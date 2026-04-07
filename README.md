# Context-Cut

> A context-aware sales agent that demonstrates three approaches to context management: naive dumping, LLM chunk compression, and Qdrant vector search pruning.

**Inspired by:** *"Is a Large Context Window All You Need? Exploring Time to First Token (TTFT)-Context Size Tradeoff for Autoregressive LLMs"* by Anuran Roy, Arnab Sengupta, and Saptarshi Pani (Alchemyst AI).

---

## The Problem

The paper demonstrates that **thoughtless context dumping** into LLMs causes:
- **O(n²) attention cost** — doubling context quadruples compute
- **Context poisoning** — irrelevant context leads the agent astray
- **Unpredictable latency spikes** — cache misses cause 22s+ response times in voice applications

Their solution: **context pruning** before the LLM call reduces latency by **38.5%** and saves **99.73%** of prompt tokens.

---

## Three Modes

| Mode | Approach | What It Does | Trade-off |
|---|---|---|---|
| **Naive** | Baseline (problem) | Dumps ALL CRM context into prompt | Slow, context-poisoned |
| **Compress** | Paper's approach | Compresses each chunk individually via LLM, then combines. Supports recursive compounding: f_m^n(x) ∝ (1/m)^n | Faster, but loses detail |
| **Prune** | Alchemyst product approach | Qdrant vector search with qwen/qwen3-embedding-8b + metadata filtering + ranking | Fast AND accurate |

### Compress Mode (Paper's Approach)

Implements the paper's algorithm exactly:

1. **Split** context into N chunks: C1, C2, ..., Cn
2. **Compress each independently**: f_m(Ci) → n(Ci)/m
3. **Combine**: f_m(C1) + f_m(C2) + ... + f_m(Cn)
4. **Recursive compounding** (optional): f_m^n(x) ∝ (1/m)^n

Complexity proof from the paper:
- Naive: n(C1+C2+C3)² = 30,000² = 9 × 10⁸
- Compressed: Σn(Ci)² + n(Σf_m(Ci))² = 3 × 10⁸ + 9 × 10⁶

### Prune Mode (Context Arithmetic + Qdrant)

Implements Alchemyst's 5-stage context pipeline:

1. **Semantic Similarity Search** — Qdrant vector search with qwen/qwen3-embedding-8b (4096-dim vectors, cosine distance)
2. **Metadata Filtering** — Qdrant `Filter` with `FieldCondition` on `contact_id`, `company_id`
3. **Deduplication** — Remove superseded context pieces
4. **Ranking** — Composite score: recency (35%) + Qdrant cosine similarity (40%) + info density (25%)
5. **Dynamic Prompt Injection** — Assemble top-K optimized context

#### Group Name Scoping

Context is scoped hierarchically, matching Alchemyst's groupName architecture:

| Scope | Level | What It Includes |
|---|---|---|
| **org** | Company | Sector, revenue, contract terms, payment behavior |
| **team** | Department | All contacts in the same department |
| **user** | Individual | Personal call history, objections, sentiment trends |

The Qdrant metadata filter enforces these scopes: queries for a specific contact only retrieve context within their org → team → user hierarchy.

---

## Quick Start

```bash
# 1. Copy env template and add your keys
cp .env.example .env
# Edit .env with your OPENROUTER_API_KEY

# 2. Start Qdrant (required for prune mode)
docker compose up -d

# 3. Index context pieces into Qdrant
uv run src/index_qdrant.py

# 4. Run the interactive demo
uv run demo.py

# 5. Run the benchmark
uv run benchmark.py
```

---

## Project Structure

```
context-cut/
├── data/
│   ├── raw/                    # Downloaded Kaggle datasets
│   └── crm.db                  # Unified SQLite database
├── src/
│   ├── config.py               # Centralized configuration
│   ├── schema.py               # SQLite schema definition
│   ├── ingest.py               # Data ingestion from CSVs
│   ├── enrich.py               # AI-enriched call summaries (with retry)
│   ├── index_qdrant.py         # Index context pieces into Qdrant
│   ├── tools/
│   │   ├── crm_lookup.py       # LangChain tools: company, contact, deal lookup
│   │   └── call_history.py     # LangChain tools: call history retrieval
│   ├── context/
│   │   ├── naive.py            # Mode 1: concatenate everything
│   │   ├── compress.py         # Mode 2: chunk-level LLM compression + recursive
│   │   └── prune.py            # Mode 3: Qdrant vector search + ranking
│   └── agents/
│       └── sales_agent.py      # Shared agent logic, swaps context assembler
├── demo.py                     # Interactive CLI demo
├── benchmark.py                # Benchmark across modes and prospects
├── docker-compose.yml          # Qdrant service
├── pyproject.toml
└── README.md
```

---

## Data

Built from three real Kaggle datasets:
- **CRM + Sales + Opportunities** — 85 companies, 8,800 deals, pipeline stages
- **Synthetic B2B CRM & Marketing** — 734 companies, 5,234 employees, interactions
- **Customer Call Center Dataset** — 20 call transcripts with full conversations

Enriched with AI-generated call summaries simulating what Alchemyst's voice AI produces.

---

## Configuration

All settings in one place via environment variables:

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | (required) | Your OpenRouter API key |
| `OPENROUTER_MODEL` | `google/gemini-2.0-flash-exp:free` | LLM model for agent responses |
| `EMBEDDING_MODEL` | `qwen/qwen3-embedding-8b` | Embedding model for vector search |
| `QDRANT_HOST` | `localhost` | Qdrant server host |
| `QDRANT_PORT` | `6333` | Qdrant server port |
| `QDRANT_COLLECTION` | `context_pieces` | Qdrant collection name |

---

## License

MIT
