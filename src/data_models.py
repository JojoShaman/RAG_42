from pydantic import BaseModel, Field
import uuid


class MinimalSource(BaseModel):
    """A source location: a file and a character range."""
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """A question without a reference answer."""
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question with its reference sources and answer."""
    sources: list[MinimalSource]
    answer: str


class RagDatasetAnswered(BaseModel):
    """A dataset of questions with reference answers."""
    rag_questions: list[AnsweredQuestion]


class RagDatasetUnanswered(BaseModel):
    """A dataset of questions without reference answers."""
    rag_questions: list[UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """The sources retrieved for one question."""
    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """The sources retrieved for one question and the generated answer."""
    answer: str


class StudentSearchResults(BaseModel):
    """Search results for a whole dataset."""
    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Search results and generated answers for a whole dataset."""
    search_results: list[MinimalAnswer]
    k: int
