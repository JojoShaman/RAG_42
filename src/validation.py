from pydantic_core import ErrorDetails
from pathlib import Path


def validation_err_output(errors: list[ErrorDetails]) -> None:
    """Print pydantic validation errors, one per line.

    Args:
        errors: Errors from ValidationError.errors().
    """
    for e in errors:
        print(".".join(map(str, e['loc'])), e['msg'])


def validate_query(query: str) -> str:
    """Validate a query given on the command line.

    Numbers are converted to strings, since Fire parses them as
    numbers.

    Args:
        query: Query as received from Fire.

    Returns:
        The query, as a string.

    Raises:
        ValueError: If the query has an invalid type, is empty, or
            contains only stopwords.
    """
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
    """Validate the number of results requested.

    Args:
        k: Number of results, as received from Fire.

    Returns:
        The validated value.

    Raises:
        ValueError: If k is not a positive integer.
    """
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
    """Validate the maximum chunk size type.

    The value range is enforced by the index command.

    Args:
        max_chunk_size: Size in characters, as received from Fire.

    Returns:
        The validated value.

    Raises:
        ValueError: If the value is not an integer.
    """
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
    """Build the output file path and check that it is safe.

    Args:
        save_dir: Output directory given on the command line.
        ref_path: Input file whose name is reused for the output.
        ref: Name of the input option, used in error messages.

    Returns:
        Path of the output file.

    Raises:
        ValueError: If save_dir is a file, or if the output would
            overwrite the input file.
    """
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
