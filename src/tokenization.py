import re


def snake_case(s: str) -> list[str]:
    if s.startswith("_") or s.endswith("_"):
        return [s]
    parts: list[str] = s.split("_")
    if len(parts) > 1:
        return parts
    return []


def camel_case(s: str) -> list[str]:
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
