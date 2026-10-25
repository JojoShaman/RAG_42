from pydantic_core import ErrorDetails
from pathlib import Path


def validation_err_output(errors: list[ErrorDetails]) -> None:
    for e in errors:
        print(".".join(map(str, e['loc'])), e['msg'])


def validate_query(query: str) -> str:
    from .tokenization import STOPWORDS
    if (isinstance(query, bool) or not
            isinstance(query, (str, int, float))):
        input_type = type(query).__name__
        raise ValueError(
            f'invalid query (got a {input_type}: {query})\n'
            'Usage: search "How do I load a LoRA adapter?" --k 5')
    query = str(query)
    if not query.strip():
        raise ValueError("Query cannot be empty")
    elif all(q in STOPWORDS for q in query.split()):
        raise ValueError("Query cannot contain stopwords only")
    return query


def validate_k(k: int) -> int:
    if isinstance(k, bool):
        raise ValueError(
            "--k expects a positive integer (got no value or a boolean)")
    elif not isinstance(k, int):
        input_type = type(k).__name__
        raise ValueError(
            f'--k expects a positive integer (got a {input_type}: {k})')
    elif k <= 0:
        raise ValueError("--k must be at least 1")
    return k


def validate_max_chunk_size(max_chunk_size: int) -> int:
    if isinstance(max_chunk_size, bool):
        raise ValueError(
            "--max_chunk_size expects a positive integer"
            " (got no value or a boolean)")
    elif not isinstance(max_chunk_size, int):
        input_type = type(max_chunk_size).__name__
        raise ValueError(
            "--max_chunk_size expects a positive integer "
            f"(got a {input_type}: {max_chunk_size})")
    return max_chunk_size


def validate_output_path(save_dir: str, ref_path: str, ref: str) -> str:
    if Path(save_dir).is_file():
        raise ValueError(
            "--save_directory path should not point to a file")
    save_dir = str(
        Path(save_dir) / Path(ref_path).name)
    if Path(save_dir).resolve() == Path(ref_path).resolve():
        raise ValueError(
            "--save_directory should not have the same path as"
            f" --{ref}")
    return save_dir
