*This project has been created as part of the 42 curriculum by srosu.*

# RAG against the machine

## Description

A Retrieval-Augmented Generation system that answers questions about the vLLM 0.10.1 codebase. It indexes the repository, retrieves the most relevant snippets for a question with BM25, and generates a short, grounded answer with `Qwen/Qwen3-0.6B`. Retrieval quality is measured with recall@k. A local HTTP API (bonus) exposes search and answer generation.

## Instructions

Requirements: Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
make install        # uv sync
make lint           # flake8 + mypy
make lint-strict    # flake8 + mypy --strict
make clean          # remove caches
make run ARGS="search 'How to load a LoRA adapter?' --k 5"
make debug ARGS="..."
```

Expected layout: the vLLM repository in `data/raw/`, the datasets in `data/datasets/`. The index is written to `data/processed/`.

## Example usage

```bash
uv run python -m src index --max_chunk_size 2000
uv run python -m src search "How to load a LoRA adapter?" --k 5
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results/UnansweredQuestions --k 10
uv run python -m src answer "How to load a LoRA adapter?" --k 3
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

### Local HTTP API (bonus)

Build the index first, then start the server (the index and the model are loaded once at startup):

```bash
uv run python -m src serve --host 127.0.0.1 --port 8000
```

Routes, available in `GET` (URL parameters) and `POST` (JSON body):

| Route | Parameters | Returns |
|---|---|---|
| `/search` | `query`, `k` (default 5) | `MinimalSearchResults` |
| `/answer` | `query`, `k` (default 5, max 10) | `MinimalAnswer` |

```bash
curl "http://127.0.0.1:8000/search?query=how+to+load+a+lora+adapter&k=3"

curl -X POST http://127.0.0.1:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"query": "How to load a LoRA adapter?", "k": 3}'
```

Invalid input returns `400`, `k` above 10 on `/answer` returns `422`, and a query matching nothing returns `404`. Interactive documentation is available at `http://127.0.0.1:8000/docs`.

## System architecture

```
data/raw → loader → chunking → tokenizer → Corpus index (data/processed)
                                                  │
question → tokenizer → BM25 ranking ──────────────┘
                          │
                    top-k sources → Qwen3-0.6B → answer (JSON)
```

- `loader.py`: reads `.py`, `.md` and `.txt` files, skipping `tests/`, `examples/`, build and VCS folders.
- `chunking.py`: splits each file into zones, then into chunks of at most `max_chunk_size` characters.
- `tokenization.py`: turns text into BM25 tokens.
- `models.py`: `Chunk` and `Corpus` (token counts, document frequencies, average length), saved as JSON.
- `BM25.py`: scoring and ranking.
- `answer.py`: prompt building and generation with Qwen.
- `validation.py`: CLI input checks. `cli.py`: the Fire commands.
- `data_models.py`: the pydantic models of the subject, plus strict dataset models.
- `api.py`: the FastAPI application (bonus).

## Chunking strategy

Two strategies, both producing *zones* (logical units) with a *context*:

- **Python**: the file is parsed with `ast`. Each top-level function or class (decorators included) is a zone, with `def name` / `class Name` as context. Consecutive module-level statements (imports, constants) are grouped into one zone. Files that fail to parse become a single zone.
- **Markdown / text**: a zone starts at every heading outside code blocks. Its context is the path of its headings (`LoRA -> Serving`). Text before the first heading is its own zone; a `.txt` file is a single zone.

Zones larger than `max_chunk_size` are split on line boundaries, with a one-line overlap. A single line longer than the limit is cut into fixed slices. Offsets are character positions in the file, so every source maps exactly to the original text.

## Retrieval method

Hand-written **BM25** (k1 = 1.5, b = 0.75).

- **Tokenization**: words are extracted with `\w+`, lowercased, and identifiers are kept whole *and* split into parts (`max_model_len` → `max_model_len`, `max`, `model`, `len`; `LLMEngine` → `llmengine`, `llm`, `engine`). This matches both questions that quote an identifier and questions that paraphrase it. A short stopword list (`what`, `how`, `the`, `vllm`…) is removed.
- **Indexing**: each chunk stores its token counts (text + context) and length; the corpus stores document frequencies and the average length. Everything is computed once and saved.
- **Ranking**: for each query token, IDF = log(1 + (N − n + 0.5) / (n + 0.5)); a chunk's score is the sum of IDF × saturated term frequency, normalized by chunk length. Chunks are sorted by score and the top k are returned. A query matching nothing returns no sources.

## Performance analysis

Moulinette recall on the public datasets:

| max_chunk_size | docs @1 / @3 / @5 / @10 | code @1 / @3 / @5 / @10 |
|---|---|---|
| 2000 | 63 / 80 / **83** / 88 % | 57.6 / 72.7 / **80.8** / 87.9 % |
| 1000 | — / — / **86** / — % | — / — / **78.8** / — % |

Both thresholds (80 % docs, 50 % code at recall@5) are met. Smaller chunks help docs (sections are more precise), larger chunks help code (functions are split less).

Main improvements during development (recall@5, docs / code): first version 58 / 55.6 % → excluding `examples/` 61 / 70.7 % → tokenizing the context 62 / 78.8 % → fixing Markdown zoning 81 / 75 % → stopwords 86 / 78.8 %.

Indexing takes under a minute and searching a 100-question dataset a few seconds on a laptop CPU. Answer generation takes about 7 to 12 s per question on CPU.

## Design decisions

- **BM25 implemented by hand** instead of a library, to control tokenization and scoring.
- **`tests/` and `examples/` excluded**: they never appear in the reference sources and pushed correct chunks out of the top 5.
- **Context added to the indexed tokens** (heading path, function or class name), so every piece of a split section keeps its topic.
- **Strict dataset models** (`RagDatasetAnswered`, `RagDatasetUnanswered`) instead of the subject's union, so validation errors point to the real missing field.
- **Generation**: 3 sources by default, one-sentence answers, thinking mode disabled, and an instruction to say "I don't know" when the context lacks the answer. Questions without sources get a fixed answer and Qwen is not called. `--k` is capped at 10 for generation.
- **Lazy imports** of `torch`/`transformers`, so index and search commands start fast.
- **Robust CLI**: every input and path is validated (types, `k ≤ 0`, empty or stopword-only queries, missing files, malformed JSON, output overwriting the input); errors are printed without tracebacks.
- **API**: built with FastAPI, reusing the pydantic models and validation functions; the index and the model stay in memory between requests.

## Challenges faced

- **Missing Markdown sections**: the last section and the text before the first heading were never indexed, so files with a single heading had no chunks. Fixing it raised docs recall from 62 % to 81 %.
- **"Magnet" chunks**: some chunks matched generic words of every question; stopwords fixed it.
- **BM25 length unit**: chunk length and average length were in different units, which favored tiny chunks.
- **pydantic union errors**: validating a question against two models produced confusing errors, solved with strict models.
- **Typing `transformers` with mypy**: `generate` is not typed on `PreTrainedModel`; handled with `isinstance` checks and one targeted `type: ignore`.
- **Generation speed on CPU**: reduced by limiting the context and the answer length.

## Resources

- Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond* (2009).
- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (2020).
- Python documentation: [`ast`](https://docs.python.org/3/library/ast.html), [`re`](https://docs.python.org/3/library/re.html), [`pathlib`](https://docs.python.org/3/library/pathlib.html).
- [pydantic](https://docs.pydantic.dev/), [Python Fire](https://github.com/google/python-fire), [Hugging Face Transformers](https://huggingface.co/docs/transformers), [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B), [FastAPI](https://fastapi.tiangolo.com/).

**AI usage**: Claude was used as a tutor to explain concepts (`ast`, BM25, regular expressions, pydantic, `transformers`, HTTP APIs). All the project code was written by me.