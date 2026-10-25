import sys
from collections import defaultdict
from pydantic import ValidationError
from pathlib import Path
from tqdm import tqdm
from .BM25 import BM25
from .models import Chunk, Corpus
from .chunking import chunk_file
from .loader import load_repo
from .validation import (
    validate_query, validate_k,
    validate_max_chunk_size,
    validation_err_output,
    validate_output_path)
from .data_models import (
    AnsweredQuestion, UnansweredQuestion,
    RagDatasetAnswered, RagDatasetUnanswered,
    StudentSearchResults, MinimalSearchResults, MinimalSource,
    StudentSearchResultsAndAnswer, MinimalAnswer)

RAW_DIR = "data/raw"
INDEX_DIR = "data/processed"
INDEX_PATH = "data/processed/corpus_index.json"


class CLI:
    def index(self, max_chunk_size: int = 2000) -> None:
        try:
            _size: int = validate_max_chunk_size(max_chunk_size)
            if _size < 100 or _size > 2000:
                _size = 100 if _size < 100 else 2000
                print("--max_chunk_size must be greater or equal to 100"
                      f" and smaller or equal to 2000\nusing {_size}"
                      " for better performance")
            if not Path(RAW_DIR).is_dir():
                raise FileNotFoundError(f"'{RAW_DIR}' directory not found")
            docs = load_repo(RAW_DIR)
            if not docs:
                raise ValueError(f"'{RAW_DIR}' directory is empty")
            chunks: list[Chunk] = []
            for d in tqdm(docs, desc="Indexing"):
                chunks.extend(chunk_file(d, _size))
            corpus: Corpus = Corpus(chunks=chunks)
            Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)
            with open(INDEX_PATH, "w") as f:
                f.write(corpus.model_dump_json(indent=2))
        except (ValueError, OSError) as e:
            print(e, file=sys.stderr)

    def search(self, query: str, k: int = 10) -> None:
        try:
            query = validate_query(query)
            k = validate_k(k)
            corpus = self._load_corpus()
            if k > len(corpus.chunks):
                print(f"--k {k} is above the total number of documents\n"
                      f"using {len(corpus.chunks)}")
                k = len(corpus.chunks)
            tf = BM25(corpus=corpus, k1=1.5, b=0.75)
            res = tf.bm25_score(query, k)
            if not any(score for score, _ in res):
                raise ValueError("No relevant results found")
            for _, meta in res:
                print(meta.file_path,
                      f"[{meta.first_character_index}:"
                      f"{meta.last_character_index}]")
        except ValidationError as e:
            validation_err_output(e.errors())
        except (ValueError, OSError) as e:
            print(e, file=sys.stderr)

    def _load_corpus(self) -> Corpus:
        with open(INDEX_PATH, "r") as f:
            corpus: Corpus = Corpus.model_validate_json(f.read())
        return corpus

    def search_dataset(self, dataset_path: str,
                       save_directory: str, k: int = 10) -> None:
        try:
            k = validate_k(k)
            corpus = self._load_corpus()
            if k > len(corpus.chunks):
                print(f"--k {k} is above the total number of documents\n"
                      f"using {len(corpus.chunks)}")
                k = len(corpus.chunks)
            save_directory = validate_output_path(
                save_directory, dataset_path, "dataset_path")
            tf = BM25(corpus=corpus, k1=1.5, b=0.75)
            with open(dataset_path, "r") as f:
                dataset: RagDatasetUnanswered = (
                    RagDatasetUnanswered.model_validate_json(f.read()))
            questions: list[UnansweredQuestion] = []
            for d_set in dataset.rag_questions:
                questions.append(
                    UnansweredQuestion(
                        question_id=d_set.question_id,
                        question=d_set.question))
            msr: list[MinimalSearchResults] = []
            for q in tqdm(questions, desc="Searching sources"):
                min_source: list[MinimalSource] = []
                res = tf.bm25_score(q.question, k)
                if any(score for score, _ in res):
                    for _, meta in res:
                        min_source.append(MinimalSource(
                            file_path=meta.file_path,
                            first_character_index=meta.first_character_index,
                            last_character_index=meta.last_character_index))
                msr.append(MinimalSearchResults(
                    question_id=q.question_id,
                    question=q.question,
                    retrieved_sources=min_source))
            ssr = StudentSearchResults(search_results=msr, k=k)
            Path(save_directory).parent.mkdir(parents=True, exist_ok=True)
            with open(save_directory, "w") as f:
                f.write(ssr.model_dump_json(indent=2))
        except ValidationError as e:
            validation_err_output(e.errors())
        except (ValueError, OSError) as e:
            print(e, file=sys.stderr)

    def answer(self, query: str, k: int = 5) -> None:
        try:
            query = validate_query(query)
            k = validate_k(k)
            if k > 10:
                print(
                    f"Warning: --k {k} is above the limit for "
                    "answer generation, using the 5 best sources instead "
                    "for better performances")
                k = 5
            tf = BM25(corpus=self._load_corpus(), k1=1.5, b=0.75)
            res = tf.bm25_score(query, k)
            sources: list[tuple[str, str]] = []
            if all(not score for score, _ in res):
                raise ValueError("No relevant results found")
            from .answer import generate_answer, load_qwen
            tokenizer, model = load_qwen()
            for _, meta in res:
                sources.append((meta.file_path, meta.text))
            print(generate_answer(query, sources, tokenizer, model))
        except ValidationError as e:
            validation_err_output(e.errors())
        except (ValueError, OSError) as e:
            print(e, file=sys.stderr)

    def answer_dataset(self, student_search_results_path: str,
                       save_directory: str, k: int = 3) -> None:
        try:
            k = validate_k(k)
            if k > 10:
                print(
                    f"Warning: --k {k} is above the limit for "
                    "answer generation, using the 5 best sources instead "
                    "for better performances")
                k = 5
            save_directory = validate_output_path(
                save_directory, student_search_results_path,
                "student_search_results_path")
            from .answer import generate_answer, load_qwen
            tokenizer, model = load_qwen()
            with open(student_search_results_path, "r") as f:
                ssr = StudentSearchResults.model_validate_json(f.read())
            answers: list[MinimalAnswer] = []
            answer: str = ""
            for q in tqdm(ssr.search_results, desc="Answering"):
                query = q.question
                chunks: list[tuple[str, str]] = []
                for source in q.retrieved_sources[:k]:
                    s = source.first_character_index
                    e = source.last_character_index
                    with open(source.file_path, "r",
                              encoding="utf-8", newline="") as f:
                        chunks.append((source.file_path, f.read()[s:e]))
                answer = (
                    "No relevant information was found in the "
                    "indexed sources to answer this question."
                    if not q.retrieved_sources else
                    generate_answer(query, chunks, tokenizer, model))
                answers.append(MinimalAnswer(
                    question_id=q.question_id,
                    question=q.question,
                    retrieved_sources=q.retrieved_sources[:k],
                    answer=answer))
            ssra = StudentSearchResultsAndAnswer(search_results=answers, k=k)
            Path(save_directory).parent.mkdir(parents=True, exist_ok=True)
            with open(save_directory, "w") as f:
                f.write(ssra.model_dump_json(indent=2))
        except ValidationError as e:
            validation_err_output(e.errors())
        except (ValueError, OSError) as e:
            print(e, file=sys.stderr)

    def evaluate(self, student_search_results_path: str,
                 dataset_path: str) -> None:
        try:
            with open(student_search_results_path, "r") as f:
                ssr = StudentSearchResults.model_validate_json(f.read())
            with open(dataset_path, "r") as f:
                dataset: RagDatasetAnswered = (
                    RagDatasetAnswered.model_validate_json(f.read()))
            if not ssr.search_results or not dataset.rag_questions:
                msg = ("Search results file contains no results"
                       if not ssr.search_results
                       else "Dataset contains no questions")
                raise ValueError(msg)
            q_id: dict[str, list[MinimalSource]] = {}
            for res in ssr.search_results:
                res_id = res.question_id
                for q in dataset.rag_questions:
                    if (res_id == q.question_id and
                            isinstance(q, AnsweredQuestion)):
                        q_id[res_id] = q.sources
                        break
            recalls: defaultdict[int, list[float]] = defaultdict(list)
            for res in ssr.search_results:
                if q_id.get(res.question_id):
                    ref = q_id[res.question_id][0]
                    s2 = ref.first_character_index
                    e2 = ref.last_character_index
                    k = 42
                    for n, source in enumerate(res.retrieved_sources):
                        if source.file_path != ref.file_path:
                            continue
                        s1 = source.first_character_index
                        e1 = source.last_character_index
                        inter = min(e1, e2) - max(s1, s2)
                        intersection = inter if inter > 0 else 0
                        union = (e1 - s1) + (e2 - s2) - intersection
                        if union > 0:
                            IoU: float = intersection / union
                            if IoU > 0.05:
                                k = n + 1
                                break
                    for rec in range(1, 11):
                        recalls[rec].append(1 if rec >= k else 0)
            lq = len(dataset.rag_questions)
            recall: list[int] = [1, 3, 5, 10]
            for r in recall:
                score = round(sum(recalls[r]) / lq, 3)
                percent = round(score * 100, 1)
                print(f"recall@{r}: {score} {percent}%")
        except ValidationError as e:
            validation_err_output(e.errors())
        except (ValueError, OSError) as e:
            print(e, file=sys.stderr)
