from pathlib import Path
from collections import deque
import ast
from .tokenization import tokenizer
from .models import Chunk

Zones = list[tuple[int, int, str]]


def init_tokens(chunk: str) -> dict[str, int]:
    """Count the occurrences of each token in a text.

    Args:
        chunk: Text to tokenize.

    Returns:
        A mapping from each token to its number of occurrences.
    """
    ret: dict[str, int] = {}
    t = tokenizer(chunk)
    for token in t:
        if ret.get(token):
            ret[token] += 1
        else:
            ret[token] = 1
    return ret


def chunk_file(doc: dict[str, str], max_size: int = 2000) -> list[Chunk]:
    """Split a document into chunks of at most max_size characters.

    The file extension selects the zoning strategy (Python or
    Markdown/text). Zones larger than max_size are split further on
    line boundaries.

    Args:
        doc: Document with "path" and "text" keys.
        max_size: Maximum chunk size, in characters.

    Returns:
        The document's chunks, with their character offsets, text,
    context and token counts.
    """
    lines = doc["text"].splitlines(keepends=True)
    if not lines:
        return []
    path = doc["path"]
    ret: list[Chunk] = []
    offset: list[int] = [0]
    for ln in lines:
        offset.append(offset[-1] + len(ln))
    extension = Path(doc["path"]).suffix
    zones = EXTS[extension](lines, offset)
    checked: list[tuple[int, int, str]] = []
    for z in zones:
        if z[1] - z[0] > max_size:
            checked.extend(segment_zone(
                lines, offset, max_size,
                offset.index(z[0]), offset.index(z[1]), z[2]))
            continue
        checked.append(z)

    for first, last, context in checked:
        txt = doc["text"][first:last]
        ret.append(
            Chunk(
                file_path=path,
                first_character_index=first,
                last_character_index=last,
                text=txt,
                tokens=init_tokens(txt + context),
                context=context))
    return ret


def segment_zone(lines: list[str], offset: list[int],
                 max_size: int, start: int,
                 end: int, context: str) -> Zones:
    """Split a zone into ranges of at most max_size characters.

    Ranges end on line boundaries, with a one-line overlap between
    consecutive ranges. A single line longer than max_size is cut
    into fixed-size slices.

    Args:
        lines: File lines, with line endings kept.
        offset: Character offset of the start of each line.
        max_size: Maximum range size, in characters.
        start: Index of the first line of the zone.
        end: Index of the line following the zone.
        context: Context attached to every range.

    Returns:
        (first_character, last_character, context) tuples.
    """
    i = start
    zones: list[tuple[int, int, str]] = []
    while i < end:
        size, j = 0, i
        while j < end and len(lines[j]) + size <= max_size:
            size += len(lines[j])
            j += 1
        if j == i:
            line = lines[i]
            for k in range(0, len(line), max_size):
                bloc = line[k:k + max_size]
                zones.append(
                    (offset[i] + k, offset[i] + k + len(bloc), context))
            i += 1
            continue
        zones.append((offset[i], offset[j], context))
        if j >= end:
            break
        back = max(i + 1, j - 1)
        i = back
    return zones


def py_zones(lines: list[str], offset: list[int]) -> Zones:
    """Find the logical zones of a Python file using its AST.

    Each top-level function or class, decorators included, becomes a
    zone with a "def name" or "class Name" context. Consecutive
    module-level statements are grouped into zones with an empty
    context. A file that fails to parse becomes a single zone.

    Args:
        lines: File lines, with line endings kept.
        offset: Character offset of the start of each line.

    Returns:
        (first_character, last_character, context) tuples.
    """
    code = "".join(lines)
    zone_type = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    no_context: dict[str, int] = {}
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return txt_zones(lines, offset)
    zones: Zones = []
    for node in tree.body:
        prefix = "class " if isinstance(node, ast.ClassDef) else "def "
        context = (prefix + node.name
                   if isinstance(node, zone_type) else "")
        start = offset[node.lineno - 1]
        if isinstance(node, zone_type) and node.decorator_list:
            start = offset[node.decorator_list[0].lineno - 1]
        end = offset[node.end_lineno] if node.end_lineno else offset[-1]
        if not context:
            if "start" not in no_context:
                no_context["start"] = start
            no_context["end"] = end
            continue
        else:
            if "start" in no_context:
                zones.append((no_context["start"], no_context["end"], ""))
                no_context = {}
        zones.append((start, end, context))
    if "start" in no_context:
        zones.append((no_context["start"], no_context["end"], ""))
    return zones


def md_zones(lines: list[str], offset: list[int]) -> Zones:
    """Find the sections of a Markdown file.

    A new zone starts at every heading outside code blocks. Each
    zone's context is the path of its headings, e.g.
    "LoRA -> Serving". Text before the first heading forms its own
    zone.

    Args:
        lines: File lines, with line endings kept.
        offset: Character offset of the start of each line.

    Returns:
        (first_character, last_character, context) tuples.
    """
    in_bash: bool = False
    zones: list[tuple[int, int, str]] = []
    title: deque[tuple[int, str]] = deque()
    context: list[str] = []
    start: int = offset[0]
    for i, ln in enumerate(lines):
        if ln.startswith("```"):
            in_bash = not in_bash
            continue
        if not in_bash:
            count = len(ln) - len(ln.lstrip("#"))
            if len(ln.lstrip("#")) == len(ln.lstrip("# ")):
                count = 0
            if count:
                end = offset[i]
                context.extend([t[1].strip("#\n") for t in title])
                while title and title[-1][0] >= count:
                    title.pop()
                if start < end:
                    zones.append((start, end, " -> ".join(context)))
                context = []
                title.append((count, ln))
                start = offset[i]
    context.extend([t[1].strip("#\n") for t in title])
    zones.append((start, offset[-1], " -> ".join(context)))
    return zones


def txt_zones(lines: list[str], offset: list[int]) -> Zones:
    """Return the whole text file as a single zone.

    Args:
        lines: File lines, with line endings kept.
        offset: Character offset of the start of each line.

    Returns:
        A single (first_character, last_character, context) tuple.
    """
    return [(offset[0], offset[len(lines)], "")]


EXTS = {".py": py_zones, ".md": md_zones, ".txt": txt_zones}
