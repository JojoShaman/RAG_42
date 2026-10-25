from pathlib import Path


SKIP_DIRS = {
    ".git",
    "node_modules",
    ".venv",
    "build",
    "dist",
    "tests",
    "examples"}
EXTS = {".py", ".md", ".txt"}


def load_repo(root: str) -> list[dict[str, str]]:
    docs: list[dict[str, str]] = []
    for path in Path(root).rglob('*'):
        if not path.is_file() or path.suffix not in EXTS:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        try:
            with open(path, encoding='utf-8', newline="") as f:
                text = f.read()
        except UnicodeDecodeError:
            continue
        docs.append({"path": str(path), "text": text})
    return docs
