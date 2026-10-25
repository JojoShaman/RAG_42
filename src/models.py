from pydantic import BaseModel, PrivateAttr
from typing import Any


class Chunk(BaseModel):
    """A searchable piece of a source file.

    Attributes:
        file_path: Path of the source file, as in the corpus.
        first_character_index: Start offset in the file.
        last_character_index: End offset in the file (exclusive).
        text: Text of the chunk.
        tokens: Token counts of the text and its context.
        context: Heading path or function/class name.
    """
    file_path: str
    first_character_index: int
    last_character_index: int
    text: str
    tokens: dict[str, int]
    context: str
    _length: int = PrivateAttr(default=0)

    def model_post_init(self, __context: Any) -> None:
        """Compute the chunk length, in tokens."""
        self._length = sum(t for _, t in self.tokens.items())


class Corpus(BaseModel):
    """The indexed collection of chunks.

    Attributes:
        chunks: All chunks of the corpus.
    """
    chunks: list[Chunk]
    _df: dict[str, int] = PrivateAttr(default_factory=dict)
    _avgdl: float = PrivateAttr(default_factory=float)

    def model_post_init(self, __context: Any) -> None:
        """Compute document frequencies and the average chunk length.

        Raises:
            ValueError: If the corpus contains no chunks.
        """
        if not self.chunks:
            raise ValueError("No documents were found")
        for chunk in self.chunks:
            for token, _ in chunk.tokens.items():
                self._df[token] = self._df.get(token, 0) + 1
        total_l = sum(n._length for n in self.chunks)
        self._avgdl = total_l / len(self.chunks)
