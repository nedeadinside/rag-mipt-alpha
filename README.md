# rag-mipt-alpha

RAG pipeline for question answering over the Alfa-Bank knowledge base (Альфа-Банк x МФТИ).

Hybrid retrieval (dense + sparse embeddings fused with RRF in Qdrant), CrossEncoder reranking,
optional multi-query expansion, LLM-based relevance verification and answer generation via Ollama.

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

End-to-end answering (retrieve + verify + generate):

```bash
python -m cmd rag --questions data/questions.csv --output submission.jsonl
```

Or run the stages separately:

```bash
python -m cmd retrieve --questions data/questions.csv --output chunks.jsonl
python -m cmd generate --input chunks.jsonl --output submission.jsonl
```

Every command accepts overrides for the relevant config sections
(`--chunk-size`, `--dense-model`, `--strategy`, `--batch-size`, `--limit`, ...).

## Pipeline

1. **Ingestion** - stream documents from CSV, split with a semantic chunker (chonkie),
   embed with dense and sparse models, upsert into a local Qdrant store (for hackathon might be okay).
2. **Retrieval** - hybrid search with RRF fusion of dense and sparse prefetches,
   optional multi-query expansion via the LLM, and reranking  down to `top_kr` chunks.
3. **Verification** - the LLM judges whether retrieved fragments are relevant to the question;
   irrelevant batches get a refusal answer instead of generation.
4. **Generation** - render the prompt template with the chunk context and call the LLM
   (with retry on transient errors).

## Project layout

```
cmd/                CLI: parser, flags, config overrides
  subcommands/      ingest, retrieve, generate, rag
src/
  clients/          FastEmbed embedders, Ollama LLM, Qdrant store, CrossEncoder reranker
  ingestion/        CSV loader, semantic chunker, indexing pipeline
  retrieval/        hybrid retriever, multi-query expander
  rag/              pipeline stages: retrieve, verify, generate
  prompts/          YAML prompt registry and templates
  types/            Runtime-checkable protocols and data models
  config.py         Pydantic settings validated from settings.yaml
```

## Development

```bash
poetry run ruff check
```
