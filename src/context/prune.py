"""Pruned context assembler — intelligent retrieval with Qdrant vector search.

This implements the Alchemyst product approach (context arithmetic):
1. Semantic similarity search — Qdrant vector search with qwen/qwen3-embedding-8b
2. Metadata filtering — hard constraints (contact_id, company_id, date)
3. Deduplication — remove stale/superseded context
4. Ranking — score by recency (35%), relevance (40%), information density (25%)
5. Dynamic prompt injection — assemble optimized context

Unlike compression (which summarizes everything), this finds the right pieces
and discards the rest. Faster AND more accurate.
"""

import math
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import requests
import tiktoken
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from src.config import (
    DB_PATH,
    EMBEDDING_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    QDRANT_COLLECTION,
    QDRANT_HOST,
    QDRANT_PORT,
    TOP_K,
    WEIGHT_INFO_DENSITY,
    WEIGHT_RECENCY,
    WEIGHT_RELEVANCE,
)


def _get_conn(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _get_qdrant_client() -> QdrantClient:
    """Get Qdrant client connection."""
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def _ensure_collection(client: QdrantClient, embedding_dim: int = 4096) -> None:
    """Create Qdrant collection if it doesn't exist.

    Qwen3-Embedding-8B produces 4096-dimensional vectors.
    """
    collections = client.get_collections().collections
    if any(c.name == QDRANT_COLLECTION for c in collections):
        return

    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(
            size=embedding_dim,
            distance=Distance.COSINE,
        ),
    )


def embed_text(text: str, max_retries: int = 3) -> list[float]:
    """Generate embedding via OpenRouter embeddings API using qwen/qwen3-embedding-8b.

    Returns a list of floats representing the embedding vector.
    Raises RuntimeError if the API is unavailable after retries.
    """
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is required for embedding-based context pruning. "
            "Set it in your .env file."
        )

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        if attempt > 0:
            backoff = min(2 ** (attempt - 1), 30)
            print(f"  Embedding retry {attempt}/{max_retries} after {backoff}s")
            time.sleep(backoff)

        try:
            resp = requests.post(
                OPENROUTER_BASE_URL + "/embeddings",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://context-cut.local",
                    "X-Title": "context-cut",
                },
                json={
                    "model": EMBEDDING_MODEL,
                    "input": text,
                    "encoding_format": "float",
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["data"][0]["embedding"]
        except Exception as e:
            last_error = e
            print(f"  Embedding attempt {attempt + 1} failed: {e}")

    raise RuntimeError(
        f"Embedding API failed after {max_retries + 1} attempts: {last_error}"
    )


def _normalize_id(raw_id: str) -> str:
    """Normalize ID to valid Qdrant UUID format."""
    if raw_id.startswith(("sum_", "syn_")):
        return raw_id[4:]
    return raw_id


def index_context_pieces(db_path: Path = DB_PATH) -> int:
    """Index all context pieces from SQLite into Qdrant.

    This should be run once after data ingestion, and again whenever
    new call summaries are added. Already-indexed points are skipped.

    Returns the number of NEW points indexed.
    """
    import time as _time

    start = _time.time()
    conn = _get_conn()
    client = _get_qdrant_client()

    # Get all call summaries with context
    rows = conn.execute("""
        SELECT cs.id, cs.contact_id, cs.company_id, cs.summary,
               cs.sentiment, cs.key_topics, cs.objections_raised,
               cs.action_items, cs.next_steps, i.date
        FROM call_summaries cs
        JOIN interactions i ON i.id = cs.interaction_id
    """).fetchall()

    if not rows:
        conn.close()
        print("  No context pieces to index")
        return 0

    print(f"  Found {len(rows)} context pieces in database")

    # Check which points are already indexed
    existing_ids: set[str] = set()
    try:
        existing = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=10000,
            with_payload=False,
            with_vectors=False,
        )
        existing_ids = {str(p.id) for p in existing[0]}
        print(f"  Found {len(existing_ids)} already indexed points in Qdrant")
    except Exception:
        print("  No existing collection — will create new one")

    # Filter out already-indexed rows
    new_rows = [r for r in rows if _normalize_id(r["id"]) not in existing_ids]
    skipped = len(rows) - len(new_rows)

    if not new_rows:
        conn.close()
        print(f"  All {len(rows)} points already indexed — nothing to do")
        return 0

    print(f"  {skipped} already indexed, {len(new_rows)} new to embed")

    # Ensure collection exists — embed first new piece to get dimension
    print("  Generating first embedding to determine vector dimension...")
    first_content_parts: list[str] = []
    if new_rows[0]["summary"]:
        first_content_parts.append(f"Summary: {new_rows[0]['summary']}")
    if new_rows[0]["sentiment"]:
        first_content_parts.append(f"Sentiment: {new_rows[0]['sentiment']}")
    if new_rows[0]["key_topics"]:
        first_content_parts.append(f"Topics: {new_rows[0]['key_topics']}")
    if new_rows[0]["objections_raised"]:
        first_content_parts.append(f"Objections: {new_rows[0]['objections_raised']}")
    if new_rows[0]["action_items"]:
        first_content_parts.append(f"Actions: {new_rows[0]['action_items']}")
    if new_rows[0]["next_steps"]:
        first_content_parts.append(f"Next: {new_rows[0]['next_steps']}")
    first_content = " | ".join(first_content_parts)
    first_embedding = embed_text(first_content)
    print(f"  Vector dimension: {len(first_embedding)}")
    _ensure_collection(client, len(first_embedding))

    # Embed and build points
    points: list[PointStruct] = []
    total = len(new_rows)

    for idx, row in enumerate(new_rows, 1):
        # Build the content payload
        content_parts: list[str] = []
        if row["summary"]:
            content_parts.append(f"Summary: {row['summary']}")
        if row["sentiment"]:
            content_parts.append(f"Sentiment: {row['sentiment']}")
        if row["key_topics"]:
            content_parts.append(f"Topics: {row['key_topics']}")
        if row["objections_raised"]:
            content_parts.append(f"Objections: {row['objections_raised']}")
        if row["action_items"]:
            content_parts.append(f"Actions: {row['action_items']}")
        if row["next_steps"]:
            content_parts.append(f"Next: {row['next_steps']}")

        content = " | ".join(content_parts)

        # Generate embedding with progress logging
        print(f"  [{idx}/{total}] Embedding: {content[:60]}...")
        vector = embed_text(content)

        point_id = _normalize_id(row["id"])

        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "type": "call",
                    "contact_id": row["contact_id"],
                    "company_id": row["company_id"],
                    "content": content,
                    "date": row["date"],
                    "sentiment": row["sentiment"],
                    "summary": row["summary"],
                },
            )
        )

        # Rate limit between embedding calls
        if idx < total:
            time.sleep(0.5)

    # Upsert in batches
    batch_size = 64
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        batch_end = min(i + batch_size, len(points))
        print(f"  Upserting batch {i + 1}-{batch_end}/{len(points)}...")
        client.upsert(collection_name=QDRANT_COLLECTION, points=batch)

    conn.close()

    elapsed = _time.time() - start
    print(f"  Indexed {len(points)} new points ({skipped} skipped) in {elapsed:.1f}s")
    return len(points)


def _days_ago(date_str: str | None) -> float:
    """Calculate days ago from a date string."""
    if not date_str:
        return 999.0
    try:
        date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        delta = datetime.now() - date.replace(tzinfo=None)
        return max(delta.days, 0)
    except (ValueError, AttributeError):
        return 999.0


def _recency_score(date_str: str | None) -> float:
    """Score by recency: 1.0 for today, decaying to 0.0 over 180 days."""
    days = _days_ago(date_str)
    if days <= 0:
        return 1.0
    return 1.0 / (1.0 + math.log(1 + days))


def _info_density_score(content: str) -> float:
    """Score by information density: ratio of meaningful content to filler."""
    if not content or len(content) < 10:
        return 0.0

    meaningful = 0
    words = content.split()
    for word in words:
        if any(c.isdigit() for c in word):
            meaningful += 1
        elif word[0].isupper() and len(word) > 2:
            meaningful += 1
        elif word.lower() in [
            "sentiment:",
            "topics:",
            "objections:",
            "action:",
            "next:",
            "stage:",
            "value:",
            "summary:",
        ]:
            meaningful += 1

    length_penalty = min(1.0, 200 / max(len(words), 1))
    return min(1.0, (meaningful / max(len(words), 1)) * 10 * length_penalty)


def _deduplicate(context_pieces: list[dict[str, object]]) -> list[dict[str, object]]:
    """Remove superseded/duplicate context pieces.

    For company/contact/deal types: keep only the most recent.
    For call types: keep all (each call is unique context).
    """
    seen: dict[str, dict[str, object]] = {}

    for piece in context_pieces:
        piece_type = str(piece.get("type", ""))
        # Only deduplicate non-call pieces (company, contact, deal)
        if piece_type != "call":
            key = piece_type
            if key in seen:
                existing = seen[key]
                if _days_ago(str(piece.get("date"))) < _days_ago(
                    str(existing.get("date"))
                ):
                    seen[key] = piece
            else:
                seen[key] = piece
        else:
            # Keep all call summaries — each is unique context
            call_id = str(
                piece.get("id", piece_type + "_" + str(piece.get("date", "")))
            )
            seen[call_id] = piece

    return list(seen.values())


def assemble_pruned_context(
    company_name: str, contact_name: str
) -> tuple[str, int, int]:
    """Assemble context using Qdrant vector search and ranking.

    Stage 1: Semantic similarity search via Qdrant + qwen/qwen3-embedding-8b
    Stage 2: Metadata filtering (contact_id, company_id)
    Stage 3: Deduplication
    Stage 4: Ranking (recency 35%, relevance 40%, info density 25%)
    Stage 5: Dynamic prompt injection

    Returns:
        Tuple of (pruned_context, original_tokens, pruned_tokens)
    """
    conn = _get_conn()

    # Get company
    company = conn.execute(
        "SELECT * FROM companies WHERE name LIKE ? LIMIT 1",
        (f"%{company_name}%",),
    ).fetchone()

    if not company:
        conn.close()
        return f"Company '{company_name}' not found.", 0, 0

    company_id = company["id"]

    # Get contact
    contact = conn.execute(
        "SELECT * FROM contacts WHERE name LIKE ? AND company_id = ? LIMIT 1",
        (f"%{contact_name}%", company_id),
    ).fetchone()

    if not contact:
        conn.close()
        return f"Contact '{contact_name}' not found at {company['name']}.", 0, 0

    contact_id = contact["id"]

    # Collect context pieces
    context_pieces: list[dict[str, object]] = []

    # 1. Company facts (always relevant)
    context_pieces.append(
        {
            "type": "company",
            "contact_id": contact_id,
            "company_id": company_id,
            "date": datetime.now().isoformat(),
            "content": (
                f"Company: {company['name']} | Sector: {company['sector']} | "
                f"Industry: {company['industry']} | Size: {company['size']} | "
                f"Revenue: {company['revenue']} | Employees: {company['employees']} | "
                f"Location: {company['location']} | Contract: {company['contract_status']} | "
                f"Payment: {company['payment_behavior']}"
            ),
            "qdrant_score": 0.0,
        }
    )

    # 2. Contact facts (always relevant)
    context_pieces.append(
        {
            "type": "contact",
            "contact_id": contact_id,
            "company_id": company_id,
            "date": datetime.now().isoformat(),
            "content": (
                f"Contact: {contact['name']} | Title: {contact['job_title']} | "
                f"Department: {contact['department']} | Seniority: {contact['seniority_level']} | "
                f"Decision Maker: {bool(contact['decision_maker'])} | "
                f"Influence: {contact['influence_score']}/100 | "
                f"Preferred: {contact['preferred_contact_method']} | "
                f"Language: {contact['language']}"
            ),
            "qdrant_score": 0.0,
        }
    )

    # 3. Current deal (most recent)
    deal = conn.execute(
        "SELECT * FROM deals WHERE company_id = ? ORDER BY engage_date DESC LIMIT 1",
        (company_id,),
    ).fetchone()

    if deal:
        context_pieces.append(
            {
                "type": "deal",
                "contact_id": contact_id,
                "company_id": company_id,
                "date": deal["engage_date"] or datetime.now().isoformat(),
                "content": (
                    f"Deal: {deal['product']} | Stage: {deal['deal_stage']} | "
                    f"Value: {deal['close_value']} | Engaged: {deal['engage_date']} | "
                    f"Closed: {deal['close_date']} | Agent: {deal['sales_agent']}"
                ),
                "qdrant_score": 0.0,
            }
        )

    # 4. Stage 1: Semantic similarity search via Qdrant
    # Build query from company + contact context
    query_text = (
        f"{company['name']} {company['sector']} "
        f"{contact['name']} {contact['job_title']} {contact['department']}"
    )
    query_vector = embed_text(query_text)

    # Stage 2: Metadata filtering via Qdrant
    qdrant_client = _get_qdrant_client()
    search_results = qdrant_client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="contact_id",
                    match=MatchValue(value=contact_id),
                ),
                FieldCondition(
                    key="company_id",
                    match=MatchValue(value=company_id),
                ),
            ]
        ),
        limit=TOP_K * 2,  # Retrieve more for re-ranking
    )

    for hit in search_results.points:
        payload = hit.payload or {}
        context_pieces.append(
            {
                "type": "call",
                "contact_id": payload.get("contact_id", contact_id),
                "company_id": payload.get("company_id", company_id),
                "date": payload.get("date", ""),
                "content": payload.get("content", ""),
                "qdrant_score": hit.score,
            }
        )

    conn.close()

    # Stage 3: Deduplication
    context_pieces = _deduplicate(context_pieces)

    # Stage 4: Ranking — combine Qdrant similarity with recency and info density
    scored_pieces: list[dict[str, object]] = []
    for piece in context_pieces:
        recency = _recency_score(str(piece.get("date")))
        qdrant_sim = float(piece.get("qdrant_score", 0.0))  # type: ignore[arg-type]
        density = _info_density_score(str(piece.get("content", "")))  # type: ignore[arg-type]

        # Qdrant cosine similarity IS the relevance score
        relevance = qdrant_sim if qdrant_sim > 0 else 0.5

        final_score = (
            WEIGHT_RECENCY * recency
            + WEIGHT_RELEVANCE * relevance
            + WEIGHT_INFO_DENSITY * density
        )

        scored_pieces.append({**piece, "score": final_score})

    # Sort by score and take top-K
    scored_pieces.sort(key=lambda x: float(x["score"]), reverse=True)  # type: ignore[arg-type]
    top_pieces = scored_pieces[:TOP_K]

    # Stage 5: Dynamic prompt injection
    context_parts: list[str] = []
    context_parts.append(f"## Context for {contact['name']} at {company['name']}")
    context_parts.append(
        f"(Top {len(top_pieces)} of {len(scored_pieces)} context pieces, "
        f"ranked by Qdrant vector search + recency + density)"
    )
    context_parts.append("")

    for piece in top_pieces:
        context_parts.append(
            f"[Score: {piece['score']:.2f} | Qdrant: {piece['qdrant_score']:.3f}] "
            f"{str(piece['type']).upper()}"
        )
        context_parts.append(str(piece["content"]))
        context_parts.append("")

    pruned_context = "\n".join(context_parts)

    # Calculate token counts
    enc = tiktoken.get_encoding("cl100k_base")
    full_context_len = len(
        enc.encode(" ".join(str(p["content"]) for p in scored_pieces))
    )
    pruned_context_len = len(enc.encode(pruned_context))

    return pruned_context, full_context_len, pruned_context_len
