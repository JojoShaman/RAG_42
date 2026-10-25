import re


def snake_case(s: str) -> list[str]:
    """Split a snake_case identifier into its parts.

    Args:
        s: Identifier to split.

    Returns:
        The parts of s, [s] if it starts or ends with an underscore,
        or an empty list if it contains no underscore.
    """
    if s.startswith("_") or s.endswith("_"):
        return [s]
    parts: list[str] = s.split("_")
    if len(parts) > 1:
        return parts
    return []


def camel_case(s: str) -> list[str]:
    """Split a camelCase or PascalCase identifier into words.

    Runs of capitals are kept together, so "LLMEngine" gives
    ["LLM", "Engine"].

    Args:
        s: Identifier to split.

    Returns:
        The words of s.
    """
    ret: list[str] = []
    i = 0
    while i < len(s):
        j = i
        while j < len(s):
            if j + 1 < len(s):
                if (s[j].isupper() and
                    s[j + 1].islower() or
                    s[j].islower() and
                        s[j + 1].isupper()):
                    if j - i > 0:
                        if s[j].isupper():
                            j -= 1
                        break
            j += 1
        ret.append(s[i:j + 1])
        if j == len(s):
            break
        i = j + 1
    return ret


def process_token(token: str) -> list[str]:
    """Return the sub-words of an identifier.

    Args:
        token: Word to split.

    Returns:
        The snake_case or camelCase parts of token, or an empty list.
    """
    proccessed: list[str] = []
    if re.search(r"[A-Z_]", token):
        if "_" in token:
            splitted = snake_case(token)
            if splitted and len(splitted) > 1:
                proccessed.extend(splitted)
        elif re.search(r"[A-Z]", token):
            proccessed.extend(camel_case(token))
    return proccessed


def tokenizer(text: str) -> list[str]:
    """Tokenize a text for BM25.

    Words are extracted, compound identifiers are split, and every
    token is lowercased. Each identifier is kept whole as well as
    split into its parts. Stopwords are removed.

    Args:
        text: Text to tokenize.

    Returns:
        The tokens, duplicates included.
    """
    isolated: list[str] = re.findall(r"\w+", text)
    tokenized: list[str] = []
    for token in isolated:
        processed: list[str] = []
        to_process: list[str] = [token]
        for t in to_process:
            processed.extend(process_token(t))
            for p in processed[:]:
                if re.search(r"[_]", t):
                    if re.search(r"[A-Z]", p):
                        to_process.append(p)
                        processed.remove(p)
        if token.lower() not in STOPWORDS:
            tokenized.append(token.lower())
        tokenized.extend(
            [tkn.lower() for tkn in processed
             if tkn.lower() not in STOPWORDS])
    return tokenized


STOPWORDS = {"what", "how", "is", "the", "a", "do",
             "you", "can", "in", "of", "to", "vllm"}
