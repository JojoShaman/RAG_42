from pydantic import BaseModel, PrivateAttr
from typing import Any


class Chunk(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int
    text: str
    tokens: dict[str, int]
    context: str
    _length: int = PrivateAttr(default=0)

    def model_post_init(self, __context: Any) -> None:
        self._length = sum(t for _, t in self.tokens.items())


class Corpus(BaseModel):
    chunks: list[Chunk]
    _df: dict[str, int] = PrivateAttr(default_factory=dict)
    _avgdl: float = PrivateAttr(default_factory=float)

    def model_post_init(self, __context: Any) -> None:
        if not self.chunks:
            raise ValueError("No documents were found")
        for chunk in self.chunks:
            for token, _ in chunk.tokens.items():
                self._df[token] = self._df.get(token, 0) + 1
        total_l = sum(n._length for n in self.chunks)
        self._avgdl = total_l / len(self.chunks)
