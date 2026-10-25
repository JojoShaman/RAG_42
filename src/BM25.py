from pydantic import BaseModel
from .tokenization import tokenizer
import math
from .models import Corpus, Chunk


class BM25(BaseModel):
    corpus: Corpus
    k1: float
    b: float

    def bm25_score(self, query: str, k: int = 1) -> list[tuple[float, Chunk]]:
        score: list[tuple[float, Chunk]] = []
        freq: int = 0
        k1 = self.k1
        b = self.b
        avgdl = self.corpus._avgdl
        q = tokenizer(query)
        idf: dict[str, float] = {}
        for t in q:
            nt = self.corpus._df.get(t, 0)
            res = math.log(
                1 + (len(self.corpus.chunks) - nt + 0.5) / (nt + 0.5))
            idf[t] = res
        for chunk in self.corpus.chunks:
            _score: list[float] = []
            for t in q:
                freq = chunk.tokens.get(t, 0)
                tf = freq / (freq + k1 * (1 - b + b * (chunk._length / avgdl)))
                _score.append(tf * idf[t])
            score.append((sum(_score), chunk))
        score.sort(key=lambda x: x[0], reverse=True)
        return score[0:k]
