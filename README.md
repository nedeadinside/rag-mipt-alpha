# rag-mipt-alpha

RAG pipeline for question answering over the Alfa-Bank knowledge base (Альфа-Банк x МФТИ).

Hybrid retrieval (dense + sparse embeddings fused with RRF in Qdrant), CrossEncoder reranking,
optional multi-query expansion, LLM-based relevance verification and answer generation via Ollama.

Runs either as a single end-to-end pass or as four resumable stages
(retrieve -> rerank -> verify -> generate) exchanging JSONL artifacts.

> [!warning]
> All cmd and README writed by Claude Code coz i hate write it by myself.
> So it can be cursed.


## Requirements

- Python 3.13
- [Poetry](https://python-poetry.org/)
- A running [Ollama](https://ollama.com/) server for LLM calls (embeddings are local via FastEmbed)

## Setup

```bash
poetry install
cp example.settings.yaml settings.yaml
```

Put the input data under `data/` (not tracked by git):

- `data/websites.csv` - source documents (`web_id`, `url`, `kind`, `title`, `text`)
- `data/questions.csv` - evaluation questions (`q_id`, `query`)

## Configuration

All settings live in `settings.yaml` and are loaded with dynaconf (`src/config.py`).
Any value can be overridden with environment variables using the `RAG` prefix,
e.g. `RAG_LLM__MODEL_NAME=...`, or with CLI flags (see `python -m cmd <command> --help`).

| Section | Purpose |
|---|---|
| `ingestion` | Qdrant collection name and path, source CSV, upsert batch size |
| `chunking` | Semantic chunker parameters: chunk size, threshold, overlap method |
| `embedding` | Dense/sparse FastEmbed models, cache dir, CUDA toggle |
| `retrieval` | Prefetch limit, reranker model and device |
| `llm` | Ollama model name, base URL, sampling parameters |
| `rag` | `top_k` / `top_kr`, search strategy, prompt template names |

## Usage

Index the knowledge base:

```bash
python -m cmd ingest
```

End-to-end answering in one pass (retrieve + rerank + verify + generate):

```bash
python -m cmd rag --questions data/questions.csv --output submission.jsonl
```

Or run the four stages separately, each reading the previous stage's JSONL:

```bash
python -m cmd retrieve --questions data/questions.csv --output chunks.jsonl
python -m cmd rerank   --input chunks.jsonl   --output reranked.jsonl
python -m cmd verify   --input reranked.jsonl --output verified.jsonl
python -m cmd generate --input verified.jsonl --output submission.jsonl
```

The staged runners (`src/../hack_optimization`) batch LLM/reranker calls and append output
line by line, so a stage interrupted mid-run resumes from its existing output: already-processed
keys are skipped. Re-running a stage continues where it stopped; delete the output to start over.

Every command accepts overrides for the relevant config sections
(`--chunk-size`, `--dense-model`, `--strategy`, `--batch-size`, `--limit`, ...);
see `python -m cmd <command> --help` for the flags exposed per stage.

## Pipeline

1. **Ingestion** - stream documents from CSV, split with a semantic chunker (chonkie),
   embed with dense and sparse models, upsert into a local Qdrant store (for hackathon might be okay).
2. **Retrieval** - hybrid search with RRF fusion of dense and sparse prefetches,
   optionally widened with multi-query expansion via the LLM, recalling `top_k` candidates.
3. **Reranking** - CrossEncoder scores the candidates against the query and keeps the
   `top_kr` best chunks.
4. **Verification** - the LLM judges whether the kept fragments are relevant to the question;
   irrelevant questions get a refusal answer instead of generation.
5. **Generation** - render the prompt template with the chunk context and call the LLM
   (with retry on transient errors).

In the one-pass `rag` command these run per batch in memory; the staged commands persist each
step to JSONL and can be resumed independently.

## Project layout

```
cmd/                CLI: parser, per-section flag groups, question CSV reader
  subcommands/      ingest, retrieve, rerank, verify, generate, rag
src/
  clients/          FastEmbed embedders, Ollama LLM, Qdrant store, CrossEncoder reranker
  ingestion/        CSV loader, semantic chunker, indexing pipeline
  retrieval/        hybrid retriever, multi-query expander
  rag/              verifier and one-pass pipeline helpers
  prompts/          YAML prompt registry and templates
  types/            Runtime-checkable protocols and data models
  config.py         Pydantic settings validated from settings.yaml
hack_optimization/  Batched, resumable stage runners used by the staged CLI
  clients.py        Batch subclasses of the base clients (LLM, reranker, expander, verifier)
  stages/           Retrieve, rerank, verify, generate stage runners over JSONL
  records.py        Pydantic artifacts exchanged between stages
  io.py             JSONL streaming, resume-key tracking, append writer
  types.py          Batch-capable protocol extensions
```

## Development

```bash
poetry run ruff check
```
