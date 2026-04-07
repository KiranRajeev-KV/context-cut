"""Compressed context assembler — LLM-based chunk-level summarization.

This implements the paper's approach exactly:
1. Split context into individual chunks (C1, C2, C3, ...)
2. Compress each chunk independently via LLM: f_m(Ci) → n(Ci)/m
3. Combine compressed chunks: f_m(C1) + f_m(C2) + f_m(C3)
4. Optionally apply recursive compounding: f_m^n(x) ∝ (1/m)^n

From the paper:
- Naive append: Q1 = C1 + C2 + C3 → complexity ∝ 30,000² = 9 × 10⁸
- Chunk compression: 3 × 10,000² + 3,000² = 3 × 10⁸ + 9 × 10⁶
- Recursive compounding: f_m^n(x) ∝ (1/m)^n

Result: 38.5% latency reduction, 99.73% token savings.
"""

import time

import requests
import tiktoken

from src.config import (
    DB_PATH,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
)

COMPRESS_PROMPT = """You are a context compressor. Compress the following CRM context chunk into a concise summary that preserves all actionable information for a sales call.

Keep:
- Key facts (names, values, dates, stages)
- Actionable insights (objections, next steps, sentiment)
- Decision-relevant details

Discard:
- Redundant phrasing
- Generic metadata
- Superseded historical details

Compress to approximately 1/10th of the original size.

Context chunk:
{chunk}

Compressed:"""

MAX_RETRIES = 3
BASE_BACKOFF = 2.0  # seconds
MAX_BACKOFF = 30.0  # seconds


def _compress_chunk_with_retry(
    chunk_content: str, compression_ratio: int = 10, max_retries: int = MAX_RETRIES
) -> str:
    """Compress a single context chunk via LLM with exponential backoff retry.

    This is f_m(C) from the paper.
    Retries on 429 (rate limit) and 5xx errors with exponential backoff.
    Falls back to truncation only after all retries exhausted.
    """
    if not OPENROUTER_API_KEY:
        return chunk_content[: len(chunk_content) // compression_ratio]

    prompt = COMPRESS_PROMPT.format(chunk=chunk_content)

    for attempt in range(max_retries + 1):
        if attempt > 0:
            backoff = min(BASE_BACKOFF * (2 ** (attempt - 1)), MAX_BACKOFF)
            print(f"    Retry {attempt}/{max_retries} after {backoff:.1f}s backoff")
            time.sleep(backoff)

        try:
            resp = requests.post(
                OPENROUTER_BASE_URL + "/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://context-cut.local",
                    "X-Title": "context-cut",
                },
                json={
                    "model": OPENROUTER_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 500,
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            if "choices" not in data or not data["choices"]:
                raise ValueError(f"Unexpected API response: {data}")
            return data["choices"][0]["message"]["content"].strip()

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if hasattr(e, "response") else 0
            if status == 429:
                print("    Rate limited (429) — will retry")
            elif 500 <= status < 600:
                print(f"    Server error ({status}) — will retry")
            else:
                # Client error (4xx) — don't retry
                print(f"    Client error ({status}) — falling back to truncation")
                return chunk_content[: len(chunk_content) // compression_ratio]
        except requests.exceptions.Timeout:
            print("    Timeout — will retry")
        except Exception as e:
            print(f"    Unexpected error: {e} — will retry")

    # All retries exhausted
    print(f"    All {max_retries + 1} attempts failed — falling back to truncation")
    return chunk_content[: len(chunk_content) // compression_ratio]


def _collect_chunks(company_name: str, contact_name: str) -> list[dict[str, str]]:
    """Collect individual context chunks from the database."""
    import sqlite3

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    company = conn.execute(
        "SELECT * FROM companies WHERE name LIKE ? LIMIT 1",
        (f"%{company_name}%",),
    ).fetchone()

    if not company:
        conn.close()
        return []

    company_id = company["id"]

    contact = conn.execute(
        "SELECT * FROM contacts WHERE name LIKE ? AND company_id = ? LIMIT 1",
        (f"%{contact_name}%", company_id),
    ).fetchone()

    if not contact:
        conn.close()
        return []

    contact_id = contact["id"]
    chunks: list[dict[str, str]] = []

    # Chunk 1: Company facts
    chunks.append(
        {
            "type": "company",
            "content": (
                f"Company: {company['name']} | Sector: {company['sector']} | "
                f"Industry: {company['industry']} | Size: {company['size']} | "
                f"Revenue: {company['revenue']} | Employees: {company['employees']} | "
                f"Location: {company['location']} | Contract: {company['contract_status']} | "
                f"Payment: {company['payment_behavior']}"
            ),
        }
    )

    # Chunk 2: Contact facts
    chunks.append(
        {
            "type": "contact",
            "content": (
                f"Contact: {contact['name']} | Title: {contact['job_title']} | "
                f"Department: {contact['department']} | Seniority: {contact['seniority_level']} | "
                f"Decision Maker: {bool(contact['decision_maker'])} | "
                f"Influence: {contact['influence_score']}/100 | "
                f"Preferred: {contact['preferred_contact_method']} | "
                f"Language: {contact['language']}"
            ),
        }
    )

    # Chunk 3: Current deal
    deal = conn.execute(
        "SELECT * FROM deals WHERE company_id = ? ORDER BY engage_date DESC LIMIT 1",
        (company_id,),
    ).fetchone()

    if deal:
        chunks.append(
            {
                "type": "deal",
                "content": (
                    f"Deal: {deal['product']} | Stage: {deal['deal_stage']} | "
                    f"Value: {deal['close_value']} | Engaged: {deal['engage_date']} | "
                    f"Closed: {deal['close_date']} | Agent: {deal['sales_agent']}"
                ),
            }
        )

    # Chunk 4+: Each call summary as an individual chunk
    calls = conn.execute(
        """SELECT cs.summary, cs.sentiment, cs.key_topics, cs.objections_raised,
                  cs.action_items, cs.next_steps, i.date
           FROM call_summaries cs
           JOIN interactions i ON i.id = cs.interaction_id
           WHERE cs.contact_id = ?
           ORDER BY i.date DESC""",
        (contact_id,),
    ).fetchall()

    for call in calls:
        content_parts: list[str] = []
        if call["summary"]:
            content_parts.append(f"Summary: {call['summary']}")
        if call["sentiment"]:
            content_parts.append(f"Sentiment: {call['sentiment']}")
        if call["key_topics"]:
            content_parts.append(f"Topics: {call['key_topics']}")
        if call["objections_raised"]:
            content_parts.append(f"Objections: {call['objections_raised']}")
        if call["action_items"]:
            content_parts.append(f"Actions: {call['action_items']}")
        if call["next_steps"]:
            content_parts.append(f"Next: {call['next_steps']}")

        chunks.append(
            {
                "type": "call",
                "content": " | ".join(content_parts),
            }
        )

    conn.close()
    return chunks


def assemble_compressed_context(
    company_name: str,
    contact_name: str,
    compression_ratio: int = 10,
    recursive_passes: int = 1,
) -> tuple[str, int, int]:
    """Compress context using the paper's chunk-level approach.

    Algorithm:
    1. Split context into N chunks: C1, C2, ..., Cn
    2. Compress each chunk: f_m(Ci) → n(Ci)/m
    3. Combine: f_m(C1) + f_m(C2) + ... + f_m(Cn)
    4. Optionally apply recursive passes: f_m^n(x) ∝ (1/m)^n

    Args:
        company_name: Company to look up
        contact_name: Contact to look up
        compression_ratio: m value (default 10 = 1/10th size)
        recursive_passes: n value for f_m^n compounding (default 1)

    Returns:
        Tuple of (compressed_context, original_tokens, compressed_tokens)
    """
    print("  Collecting context chunks...")
    chunks = _collect_chunks(company_name, contact_name)

    if not chunks:
        return f"No context found for {contact_name} at {company_name}.", 0, 0

    print(f"  Found {len(chunks)} chunks to compress")

    enc = tiktoken.get_encoding("cl100k_base")

    # Calculate original token count
    original_tokens = sum(len(enc.encode(c["content"])) for c in chunks)
    print(f"  Original context: {original_tokens} tokens across {len(chunks)} chunks")

    # Stage 1: Compress each chunk independently — f_m(Ci)
    compressed_chunks: list[dict[str, str]] = []
    for i, chunk in enumerate(chunks):
        print(f"  [{i + 1}/{len(chunks)}] Compressing {chunk['type']}...")
        compressed_content = _compress_chunk_with_retry(
            chunk["content"], compression_ratio
        )
        compressed_tokens = len(enc.encode(compressed_content))
        original_chunk_tokens = len(enc.encode(chunk["content"]))
        reduction = (
            (original_chunk_tokens - compressed_tokens) / max(original_chunk_tokens, 1)
        ) * 100
        print(
            f"    {original_chunk_tokens} → {compressed_tokens} tokens ({reduction:.0f}% reduction)"
        )
        compressed_chunks.append(
            {
                "type": chunk["type"],
                "content": compressed_content,
            }
        )
        # Rate limit between chunk compressions
        if i < len(chunks) - 1:
            time.sleep(1.0)

    # Stage 2: Combine compressed chunks
    combined = "\n\n".join(
        f"## {c['type'].upper()}\n{c['content']}" for c in compressed_chunks
    )

    # Stage 3: Recursive compounding — f_m^n
    # Default is 1 pass (no recursion). Set recursive_passes > 1 for compounding.
    current = combined
    for pass_num in range(1, recursive_passes):
        print(f"  Recursive compression pass {pass_num + 1}/{recursive_passes}...")
        current = _compress_chunk_with_retry(current, compression_ratio)

    compressed_tokens = len(enc.encode(current))

    total_reduction = (
        (original_tokens - compressed_tokens) / max(original_tokens, 1)
    ) * 100
    print(
        f"  Final: {original_tokens} → {compressed_tokens} tokens ({total_reduction:.0f}% reduction)"
    )

    return current, original_tokens, compressed_tokens
