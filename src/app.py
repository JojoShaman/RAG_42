from fastapi import FastAPI, HTTPException
from .BM25 import BM25
from pydantic import BaseModel
from .data_models import (
    MinimalSearchResults,
    MinimalSource,
    UnansweredQuestion,
    MinimalAnswer)
from transformers import PreTrainedTokenizerBase, PreTrainedModel
from .validation import validate_query, validate_k


class SearchRequest(BaseModel):
    query: str
    k: int


class AnswerRequest(BaseModel):
    query: str
    k: int


def build_app(
        bm25: BM25,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel) -> FastAPI:
    app = FastAPI()

    @app.post("/search")
    def search_post(req: SearchRequest) -> MinimalSearchResults:
        return search_route(req)

    @app.get("/search")
    def search_get(query: str, k: int = 5) -> MinimalSearchResults:
        return search_route(SearchRequest(query=query, k=k))

    def search_route(req: SearchRequest) -> MinimalSearchResults:
        try:
            query = validate_query(req.query)
            k = validate_k(req.k)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        min_source: list[MinimalSource] = []
        q = UnansweredQuestion(question=query)
        res = bm25.bm25_score(query, k)
        if not any(score for score, _ in res):
            raise HTTPException(
                status_code=404,
                detail="No relevant results found for this query")
        for _, meta in res:
            min_source.append(MinimalSource(
                file_path=meta.file_path,
                first_character_index=meta.first_character_index,
                last_character_index=meta.last_character_index))
        return MinimalSearchResults(
            question_id=q.question_id,
            question=q.question,
            retrieved_sources=min_source)

    @app.post("/answer")
    def answer_post(req: AnswerRequest) -> MinimalAnswer:
        return answer_route(req)

    @app.get("/answer")
    def answer_get(query: str, k: int = 5) -> MinimalAnswer:
        return answer_route(AnswerRequest(query=query, k=k))

    def answer_route(req: AnswerRequest) -> MinimalAnswer:
        try:
            query = validate_query(req.query)
            k = validate_k(req.k)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if k > 10:
            raise HTTPException(status_code=422, detail=(
                f"Warning: k {k} is above the limit for "
                "answer generation"))
        q = UnansweredQuestion(question=query)
        res = bm25.bm25_score(query, k)
        if not any(score for score, _ in res):
            raise HTTPException(
                status_code=404,
                detail="No relevant results found for this query")
        sources: list[tuple[str, str]] = []
        retrieved: list[MinimalSource] = []
        for _, meta in res:
            sources.append((meta.file_path, meta.text))
            retrieved.append(MinimalSource(
                file_path=meta.file_path,
                first_character_index=meta.first_character_index,
                last_character_index=meta.last_character_index))
        from .answer import generate_answer
        ret = MinimalAnswer(
            question_id=q.question_id,
            question=q.question,
            retrieved_sources=retrieved,
            answer=generate_answer(q.question, sources, tokenizer, model))
        return ret
    return app
