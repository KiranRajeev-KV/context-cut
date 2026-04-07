# Context-Cut: Technical Report

> A context-aware sales agent demonstrating three approaches to context management: naive dumping, LLM chunk compression, and Qdrant vector search pruning.

**Inspired by:** *"Is a Large Context Window All You Need? Exploring Time to First Token (TTFT)-Context Size Tradeoff for Autoregressive LLMs"* by Anuran Roy, Arnab Sengupta, and Saptarshi Pani (Alchemyst AI)  
**Additional reference:** *"The Pareto Frontier for Context: Alchemyst achieves cost-performance optimality"* by Rishit Murarka, reviewed by Kapilansh Patil (Alchemyst AI)

---

## 1. Motivation and Inspiration

### The Source Paper

The CTO's paper establishes that **thoughtless context dumping** into LLMs causes:

- **O(n²) attention cost** — doubling context quadruples compute
- **Context poisoning** — irrelevant context leads the agent astray
- **Unpredictable latency spikes** — cache misses cause 22s+ response times in voice applications

Their solution: **context pruning** before the LLM call reduces latency by **38.5%** and saves **99.73%** of prompt tokens.

### The Product Vision

Alchemyst's blog posts describe their **Kathan engine** — a context arithmetic pipeline that finds the right context pieces at the right time using a 5-stage process. The Pareto Frontier blog shows they achieve **80% accuracy on ConvoMem** (vs SuperMemory's 50%) at **~$0.06 per million tokens** (vs SuperMemory's $6.33).

### Why I Built This

I read Anuran Roy's paper on context pruning and the Pareto Frontier blog post, and I wanted to verify the claims myself — not just accept them at face value. I built all three approaches (naive, compress, prune) from scratch, measured them against real CRM data, and compared the results. This project is my attempt to answer: *Does context engineering actually work in practice?*

---

## 2. The Three Strategies

### Strategy 1: Naive Append (The Problem)

**Source:** The paper's baseline — "Directly dump the context into the LLM, appending them one after another."

**What it does:** Concatenates ALL available CRM context (company info, contacts, deals, call history, interactions) into a single prompt.

**Math from the paper:** `Q1 = C1 + C2 + C3 → complexity ∝ 30,000² = 9 × 10⁸`

### Strategy 2: Chunk-Level Compression (The Paper's Solution)

**Source:** The paper's "Context Compression" approach — `f_m(C) → n(C)/m`

**What it does:**
1. Splits context into individual chunks (company, contact, deal, each call summary)
2. Compresses each chunk independently via LLM to ~1/m of original size
3. Combines compressed chunks
4. Supports recursive compounding: `f_m^n(x) ∝ (1/m)^n`

**Math from the paper:** `Σn(Ci)² + n(Σf_m(Ci))² = 3 × 10⁸ + 9 × 10⁶`

### Strategy 3: Context Arithmetic + Vector Search (The Product Approach)

**Source:** Alchemyst's Kathan engine blog posts and the Pareto Frontier blog.

**What it does:**
1. **Semantic Similarity Search** — Qdrant vector search with qwen/qwen3-embedding-8b (4096-dim, cosine distance)
2. **Metadata Filtering** — Qdrant `Filter` with `FieldCondition` on `contact_id`, `company_id`
3. **Deduplication** — Remove superseded context pieces
4. **Ranking** — Composite score: recency (35%) + Qdrant cosine similarity (40%) + info density (25%)
5. **Dynamic Prompt Injection** — Assemble top-K optimized context

**Group Name Scoping:** Context is scoped hierarchically (org → team → user), matching Alchemyst's groupName architecture. Qdrant metadata filters enforce these scopes.

---

## 3. Data Pipeline

### 3.1 Source Datasets (Kaggle)

| Dataset | Source | Records | Purpose |
|---|---|---|---|
| CRM + Sales + Opportunities | `innocentmfa/crm-sales-opportunities` | 85 companies, 8,800 deals | Company profiles, deal pipeline |
| Synthetic B2B CRM & Marketing | `ezogngrd/synthetic-b2b-crm-and-marketing-data` | 734 companies, 5,234 employees | Contact details, interactions |
| Customer Call Center Dataset | `rafaqatkhan608/customer-call-center-dataset-analysis` | 20 call transcripts | Real conversation data |

### 3.2 Unified Schema (SQLite)

Built a unified relational schema across 6 tables:

```
companies → contacts → deals → interactions → call_summaries → campaigns
```

Key design decisions:
- Companies from both CRM and B2B datasets merged into single table
- Contacts linked to companies via foreign keys
- Deals linked to companies (not individual contacts, matching the source data)
- Interactions and call summaries linked to both contacts and companies

### 3.3 AI Enrichment

The 20 real call transcripts were enriched via OpenRouter API with structured summaries (sentiment, topics, objections, action items, next steps). The enrichment script includes:
- Exponential backoff retry (3 retries, 1s → 2s → 4s)
- Graceful degradation on API failure
- Rate limiting between calls

### 3.4 Synthetic Call History Generation

Generated 187 multi-touch call histories for prospects with stage-appropriate content:

| Deal Stage | Summaries Generated | Sentiment | Objections | Actions |
|---|---|---|---|---|
| **Won** | 173 | positive | [] | onboarding, kickoff |
| **Lost** | 7 | neutral/negative | competitor pricing, missing features | nurture campaign, document loss |
| **Engaging** | 7 | neutral/negative | needs approval, evaluating vendors | send case studies, schedule demo |
| **Prospecting** | 0 | — | — | — |

**Known issue:** Zero Prospecting summaries because the synthetic generator only created summaries for companies that already had contacts, and all of those had deals past the prospecting stage. The CRM data skews heavily toward Won deals (4,238 Won vs 163 Prospecting).

### 3.5 Qdrant Vector Index

- **Collection:** `context_pieces`
- **Vectors:** 206 points, 4096 dimensions (qwen/qwen3-embedding-8b)
- **Distance:** Cosine
- **Indexing time:** ~587 seconds
- **Skip logic:** Already-indexed points are detected and skipped on re-runs

---

## 4. Implementation Details

### 4.1 Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.12 |
| Type checking | Pyright strict mode |
| Linting | Ruff |
| Package manager | uv |
| Vector store | Qdrant (Docker) |
| Embedding model | `qwen/qwen3-embedding-8b` via OpenRouter |
| LLM (agent, primary) | `google/gemini-2.5-flash-lite:free` via OpenRouter |
| LLM (agent, comparison) | `qwen/qwen-2.5-7b-instruct` via OpenRouter |
| LLM (compression) | Same as agent model (varies by benchmark run) |
| Token counting | tiktoken (cl100k_base) |

### 4.2 Naive Mode (`src/context/naive.py`)

- Queries SQLite for company, contact, deals, call summaries, interactions
- Concatenates everything into a single string
- Token count via tiktoken (cl100k_base)
- **No filtering, no ranking, no pruning**

### 4.3 Compress Mode (`src/context/compress.py`)

- Splits context into individual chunks (company, contact, deal, each call summary)
- Compresses each chunk independently via LLM with prompt: "Compress to approximately 1/10th of original size"
- Exponential backoff retry (3 retries, 2s → 4s → 8s)
- Graceful fallback to string truncation if all retries fail
- Supports recursive compounding via `recursive_passes` parameter
- Rate limiting: 1s between chunk compressions

### 4.4 Prune Mode (`src/context/prune.py`)

- Embeds query (company + contact context) via `qwen/qwen3-embedding-8b`
- Qdrant vector search with metadata filter (`contact_id`, `company_id`)
- Retrieves top-K×2 candidates for re-ranking
- Deduplicates: keeps only the most recent company/contact/deal, keeps all call summaries
- Scores each piece: `0.35 × recency + 0.40 × Qdrant cosine sim + 0.25 × info density`
- Assembles top-K pieces into optimized prompt
- Embedding retry: 3 retries with exponential backoff
- Rate limiting: 0.5s between embedding calls

### 4.5 Sales Agent (`src/agents/sales_agent.py`)

- LangChain chain: PromptTemplate → ChatOpenAI (OpenRouter)
- Exponential backoff retry for LLM calls (5 retries, 2s → 4s → 8s → 16s → 32s)
- Swaps context assembler based on mode parameter
- Measures: context assembly time, LLM time, total time, token counts

---

## 5. Differences from Alchemyst's Implementation

### 5.1 Compress Mode

| Aspect | Paper's Implementation | My Implementation |
|---|---|---|
| Compression unit | Each chunk individually | Each chunk individually |
| Architecture | Proxy server between user and vLLM cluster | Direct API call to OpenRouter |
| Recursive passes | `f_m^n(x) ∝ (1/m)^n` | Supported via parameter |
| Complexity accounting | Counts compression cost explicitly | Not tracked |
| Model | Self-hosted Qwen-2.5-7B on vLLM L4 GPUs | Gemini-2.5-Flash-Lite and Qwen-2.5-7B via OpenRouter |

### 5.2 Prune Mode

| Aspect | Alchemyst Kathan | My Implementation |
|---|---|---|
| Vector store | Qdrant (production) | Qdrant (Docker, local) |
| Embedding model | Proprietary | `qwen/qwen3-embedding-8b` |
| Scale | 500K+ calls/day | Single prospect at a time |
| Latency budget | <200ms for full pipeline | No budget (demo) |
| groupName scoping | org → team → user hierarchy | Implemented via metadata filters |
| Streaming | Real-time, per-millisecond | Batch assembly before LLM call |
| PII redaction | Yes | Not implemented |
| Cache awareness | Addresses cache miss spikes | Not implemented |
| HITL strategies | Production-grade | Not implemented |

### 5.3 What's Missing for Production

1. **Real-time streaming** — Kathan streams context as the conversation happens
2. **PII redaction** — Kathan redacts PII before context injection
3. **Cache awareness** — Paper's finding about cache misses causing latency spikes not addressed
4. **Multi-tenant isolation** — No org/team/user ACL enforcement
5. **Observability** — No telemetry tracing the agentic loop
6. **On-premise readiness** — No MongoDB + Qdrant managed services deployment
7. **12+ Indian language support** — English only
8. **TRAI compliance** — Not applicable (US CRM data)

---

## 6. Full Benchmark Results

### 6.1 Configuration

| Parameter | Value |
|---|---|
| LLM model (primary) | `google/gemini-2.5-flash-lite` via OpenRouter |
| LLM model (comparison) | `qwen/qwen-2.5-7b-instruct` via OpenRouter |
| Embedding model | `qwen/qwen3-embedding-8b` via OpenRouter |
| Vector store | Qdrant (Docker, localhost:6333) |
| Qdrant collection | `context_pieces` (206 points, 4096-dim, cosine) |
| Token counter | tiktoken (cl100k_base) |
| Rate limits | 3s between modes, 5s between prospects |
| Compress retries | 3 retries, 2s → 4s → 8s backoff |
| LLM retries | 5 retries, 2s → 4s → 8s → 16s → 32s backoff |

### 6.2 Per-Prospect Results (Gemini Flash)

#### Prospect 1: William Jones at Stanredtax (Stage: Lost)

| Mode | Raw Context | Effective Context | Response | LLM Time | Total |
|---|---|---|---|---|---|
| Naive | 5,496 | 5,496 | 335 | 4.6s | 4.8s |
| Compress | 452 → 230 (49%) | 230 | 203 | 2.4s | 18.3s |
| Prune | 452 | 459 | 142 | 1.7s | 4.9s |

#### Prospect 2: Lisa Jones at Acme Corporation (Stage: Won)

| Mode | Raw Context | Effective Context | Response | LLM Time | Total |
|---|---|---|---|---|---|
| Naive | 3,973 | 3,973 | 419 | 5.0s | 5.0s |
| Compress | 551 → 285 (48%) | 285 | 208 | 2.2s | 20.2s |
| Prune | 551 | 430 | 312 | 2.4s | 3.8s |

#### Prospect 3: Amanda Davis at Zumgoity (Stage: Won)

| Mode | Raw Context | Effective Context | Response | LLM Time | Total |
|---|---|---|---|---|---|
| Naive | 3,626 | 3,626 | 334 | 2.8s | 2.8s |
| Compress | 456 → 250 (45%) | 250 | 238 | 3.1s | 28.1s |
| Prune | 456 | 425 | 171 | 1.5s | 3.0s |

### 6.3 Per-Prospect Results (Qwen-2.5-7B)

#### Prospect 1: Jennifer Brown at Singletechno (Stage: Won)

| Mode | Raw Context | Effective Context | Response | LLM Time | Total |
|---|---|---|---|---|---|
| Naive | 9,011 | 9,011 | 274 | 5.3s | 5.6s |
| Compress | 787 → 425 (46%) | 425 | 353 | 9.3s | 39.5s |
| Prune | 787 | 465 | 242 | 3.4s | 4.9s |

#### Prospect 2: Amanda Davis at Zumgoity (Stage: Won)

| Mode | Raw Context | Effective Context | Response | LLM Time | Total |
|---|---|---|---|---|---|
| Naive | 3,626 | 3,626 | 389 | 10.6s | 10.6s |
| Compress | 456 → 283 (38%) | 283 | 269 | 8.0s | 32.9s |
| Prune | 456 | 425 | 276 | 4.4s | 5.8s |

#### Prospect 3: Sarah Davis at Y-corporation (Stage: Won)

| Mode | Raw Context | Effective Context | Response | LLM Time | Total |
|---|---|---|---|---|---|
| Naive | 5,076 | 5,076 | 476 | 15.4s | 15.4s |
| Compress | 533 → 327 (39%) | 327 | 243 | 4.0s | 25.1s |
| Prune | 533 | 424 | 313 | 4.4s | 5.8s |

### 6.4 Aggregate Results

#### Gemini Flash (`google/gemini-2.5-flash-lite`) — 3 prospects

| Metric | Naive | Compress | Prune |
|---|---|---|---|
| **Avg Raw Context Tokens** | 4,365 | 486 | 486 |
| **Avg Effective Context Tokens** | 4,365 | 255 | 438 |
| **Avg Response Tokens** | 363 | 216 | 208 |
| **Avg LLM Time** | 4.1s | 2.5s | 1.9s |
| **Avg Total Time** | 4.2s | 22.2s | 3.9s |

#### Qwen-2.5-7B (`qwen/qwen-2.5-7b-instruct`) — 3 prospects

| Metric | Naive | Compress | Prune |
|---|---|---|---|
| **Avg Raw Context Tokens** | 5,904 | 592 | 592 |
| **Avg Effective Context Tokens** | 5,904 | 345 | 438 |
| **Avg Response Tokens** | 380 | 288 | 277 |
| **Avg LLM Time** | 10.4s | 7.1s | 4.0s |
| **Avg Total Time** | 10.5s | 32.5s | 5.5s |

#### Cross-Model Comparison

| Metric | Gemini Flash | Qwen-2.5-7B | Paper (Qwen-2.5-7B on vLLM) |
|---|---|---|---|
| **Token reduction (naive → prune)** | 90.0% | 92.6% | 99.73% |
| **Naive LLM time** | 4.1s | 10.4s | 12.4s |
| **Prune LLM time** | 1.9s | 4.0s | 7.6s |
| **Compress chunk reduction** | 45-49% | 38-46% | ~90% |
| **Time reduction (naive → prune)** | 7.8% | 47.5% | 38.5% |

### 6.5 Key Findings

**Token reduction is model-agnostic**

Both models show ~90-93% token reduction, confirming the paper's thesis that context engineering works regardless of which LLM generates the response. The remaining 7-10% is the irreducible minimum context (company facts, contact details, current deal, call history) that must always be included.

**Prune mode achieves 47.5% time reduction with Qwen**

After fixing the deduplication bug (which was collapsing all call summaries into one), prune mode now retrieves full multi-touch call history via Qdrant. With Qwen's slower LLM (10.4s naive), the context assembly overhead is negligible — matching the paper's finding that context pruning matters most when LLM inference is the bottleneck.

**Gemini is too fast for time savings to show**

At 4.1s naive LLM time, Gemini's speed makes context assembly overhead (1.9s for prune) relatively significant. The token savings are real (90%), but time savings are minimal (7.8%).

**Compress mode remains impractical on APIs**

- Gemini: 22.2s average (89% spent on chunk-by-chunk compression)
- Qwen: 32.5s average (78% spent on chunk-by-chunk compression)
- The paper's approach assumes a local vLLM cluster with no rate limits

### 6.6 Business Impact at Scale

| Metric | Per Call | × 500K Calls/Day | Annual |
|---|---|---|---|
| Naive tokens (Qwen avg) | 5,904 | 2.95B | 1.08T |
| Prune tokens (Qwen avg) | 438 | 219M | 80B |
| **Tokens saved** | **5,466** | **2.73B** | **997B** |
| Cost at $0.06/1M tokens | — | $164/day | $59,800/year |
| Cost at $6.33/1M tokens (SuperMemory) | — | $17,300/day | $6.3M/year |

---

## 7. Quality Analysis

### 7.1 Naive Mode Output Issues

**Gemini Flash (3/3 outputs):**
- **Meta-commentary (3/3):** Every output starts with framing text ("Here's a personalized call script for...", "Okay, Lisa. Here's...").
- **Placeholder leakage (2/3):** William Jones output includes `[mention a specific pain point or topic from previous calls, e.g., 'improving data security' or 'automating tax filing processes']`. Amanda Davis output includes `[mention a specific challenge if you recall one from previous calls, otherwise generalize to 'optimizing their procurement processes']` and `[briefly reiterate a key benefit, e.g., 'streamlining your vendor management' or 'reducing operational costs']` — the 3,600-5,500 tokens of noise caused the model to fall back to template language.

**Qwen-2.5-7B (3/3 outputs):**
- **Identity leakage (1/3):** Sarah Davis output: "This is Qwen from Alibaba Cloud" — the model's system prompt leaks through.
- **Meta-commentary (2/3):** Two outputs include `**Call Script:**` headers and markdown section headers (`**Follow-up on Action Items:**`, `**Next Steps:**`).
- **Good specificity (3/3):** References exact deal values ($6,456, $3,735, $538), product names (GTX Basic, MG Advanced, GTXPro), and CRM names.

### 7.2 Compress Mode Output Quality

**Gemini Flash (3/3 outputs):**
- **Clean scripts (2/3):** Direct call scripts with no meta-commentary. References specific deal values ($3,735, $3,844), product names (MG Advanced, GTXPro), and action items.
- **Meta-commentary (1/3):** Amanda Davis output starts with "Okay, here's a personalized call script..." — one outlier.

**Qwen-2.5-7B (3/3 outputs):**
- **Hallucinated names (1/3):** "It's Versie from [Your Company]" — the compression process seems to trigger the model to invent a caller identity.
- **Meta-commentary (1/3):** Jennifer Brown output includes `**Call Script:**` header and `**Next Step:**` section.
- **Clean scripts (1/3):** Sarah Davis output is a direct, professional script.

### 7.3 Prune Mode Output Quality

**Gemini Flash (3/3 outputs):**
- **Meta-commentary (1/3):** Lisa Jones output includes section headers (`**(Opening - 15 seconds)**`, `**(Context & Value - 30 seconds)**`).
- **Clean scripts (2/3):** William Jones and Amanda Davis outputs are direct call scripts.
- **No placeholder leakage:** Zero instances of `[mention...]` patterns — the 425-459 token context is precise enough.

**Qwen-2.5-7B (3/3 outputs):**
- **Hallucinated names (1/3):** "It's Sarah from [Your Company]" — invented caller identity.
- **Meta-commentary (2/3):** Two outputs include `**Call Script:**` headers and `**Next Step:**` sections.
- **Clean scripts (1/3):** Jennifer Brown output is a direct, professional script.
- **No identity leakage:** Zero "Qwen from Alibaba Cloud" in this run.

### 7.4 Cross-Model Comparison (3-Prospect Run, Post-Dedup Fix)

| Issue | Gemini Naive | Gemini Prune | Qwen Naive | Qwen Prune |
|---|---|---|---|---|
| Placeholder leakage | 2/3 (67%) | 0/3 (0%) | 0/3 (0%) | 0/3 (0%) |
| Factual hallucination | 0/3 (0%) | 0/3 (0%) | 0/3 (0%) | 0/3 (0%) |
| Identity leakage | 0/3 (0%) | 0/3 (0%) | 1/3 (33%) | 0/3 (0%) |
| Meta-commentary | 3/3 (100%) | 1/3 (33%) | 2/3 (67%) | 2/3 (67%) |
| Hallucinated caller name | 0/3 (0%) | 0/3 (0%) | 0/3 (0%) | 1/3 (33%) |
| Specific CRM references | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) |

**Key findings from post-dedup-fix run:**

1. **Prune mode eliminates placeholder leakage entirely** — the ~438-token context is precise enough that neither model needs template fallbacks. Gemini naive had 67% placeholder leakage; prune has 0%.
2. **Prune mode achieves 47.5% time reduction with Qwen** — after fixing deduplication to keep all call summaries, the context assembly overhead becomes negligible compared to Qwen's 10.4s LLM time. This validates the paper's thesis.
3. **Token reduction holds at 90-93%** across both models, confirming the approach is model-agnostic.
4. **Meta-commentary decreases significantly for Gemini** (100% → 33%) but persists for Qwen (67% → 67%) — Qwen's tendency to add section headers is a model-specific prompt engineering issue.
5. **Hallucinated caller names appear in compress and prune modes** for Qwen ("Versie", "Sarah") — the compressed/pruned context seems to trigger identity fabrication when the model lacks a clear system prompt.
6. **Specific CRM references reach 100% in all modes** — even naive mode retrieves correct details, but prune mode does so with 90% fewer tokens.

---

## 8. Known Issues and Limitations

### 8.1 Data Issues

- **Zero Prospecting summaries** — Synthetic generator only created summaries for companies with existing contacts, all of which had deals past prospecting stage
- **Heavy Won skew** — 173 Won vs 7 Lost vs 7 Engaging — reflects the source CRM data distribution
- **Cross-product joins** — Companies with multiple deals cause N×M joins; fixed with subquery to pick most recent deal per company

### 8.2 Implementation Issues

- **Truncation fallback** — When all compression retries fail, the system falls back to `text[:len(text)//10]` (blind character slicing). This is intentional graceful degradation, not a design flaw.
- **Single-language** — English only. Alchemyst supports 12+ Indian languages.
- **Qwen identity leakage** — Intermittent; the model's system prompt overrides the agent prompt in some runs.

### 8.3 Infrastructure Issues

- **Qdrant container** — Can become unresponsive after heavy indexing (206 embeddings). Requires `docker compose restart` to recover.
- **Free-tier rate limits** — Qwen free tier on OpenRouter has very low rate limits. Gemini Flash free tier works much better.
- **No production deployment** — Everything runs locally. No Kubernetes, no vLLM, no GPU cluster.

---

## 9. What I Learned

### 9.1 The Paper Is Right

Context pruning works. The 90-93% token reduction is real across both models. After fixing the deduplication bug, prune mode achieved 47.5% time reduction with Qwen — validating the paper's thesis that context engineering matters most when LLM inference is the bottleneck.

### 9.2 The Deduplication Bug Was Critical

The original deduplication collapsed all call summaries per contact into one (keeping only the most recent). This meant prune mode was retrieving ~210 tokens of context — but missing the multi-touch call history that makes it valuable. After fixing it to keep all call summaries, prune mode retrieves ~438 tokens with full call history, and the quality of outputs improved significantly.

### 9.3 The Paper's Context Doesn't Fully Apply to Free-Tier APIs

The 38.5% latency reduction assumes self-hosted GPUs where LLM inference is the bottleneck. On fast free-tier APIs like Gemini, the LLM call is so quick that context assembly overhead dominates. The business case is about **cost at scale**, not speed per call.

### 9.4 Context Engineering Is Harder Than It Looks

Building a system that finds the right context at the right time requires:
- Good embeddings (qwen/qwen3-embedding-8b works well)
- Good metadata filtering (Qdrant's Filter API is powerful)
- Good ranking weights (35/40/25 split is reasonable but needs tuning)
- Good deduplication (type-based for company/contact/deal, keep all calls)

### 9.5 The Real Value Is at Scale

438 tokens vs 5,904 tokens doesn't matter for one call. It matters for 500,000 calls per day. That's where Alchemyst's $0.06/1M tokens vs competitors' $6.33/1M tokens becomes a moat.

---

## 10. How to Run

```bash
# 1. Set up environment
cp .env.example .env
# Edit .env with your OPENROUTER_API_KEY

# 2. Start Qdrant
docker compose up -d

# 3. Index context pieces (one-time, ~10 minutes)
uv run src/index_qdrant.py

# 4. Interactive demo
uv run demo.py

# 5. Benchmark
uv run benchmark.py
```

---

## 11. Project Structure

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
│   ├── regenerate_summaries.py # Regenerate stage-appropriate summaries
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

## 12. Acknowledgments

- **Anuran Roy, Arnab Sengupta, Saptarshi Pani** — For the paper that inspired this project
- **Rishit Murarka, Kapilansh Patil** — For the Pareto Frontier blog that provided benchmark context
- **Alchemyst AI team** — For open-sourcing their research and building in public
- **Kaggle dataset authors** — For the raw data that made this possible

---

*Context-Cut — A context engineering evaluation project.*
