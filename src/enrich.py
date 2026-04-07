"""Enrich call transcripts with AI-generated summaries via OpenRouter."""

import json
import sqlite3
import time
from pathlib import Path

import requests

from src.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL

DB_PATH = Path(__file__).parent.parent / "data" / "crm.db"

PROMPT_TEMPLATE = """You are a sales call analyst. Given a call transcript, produce a structured summary.

Transcript:
{transcript}

Respond with ONLY valid JSON. No markdown, no explanations.

{{
  "summary": "2-3 sentence summary of the call",
  "key_topics": ["topic1", "topic2", "topic3"],
  "objections_raised": ["objection1", "objection2"],
  "sentiment": "positive|neutral|negative",
  "language": "en|es|hi|etc",
  "action_items": ["action1", "action2"],
  "next_steps": "What should happen next"
}}
"""

MAX_RETRIES = 3
BASE_BACKOFF = 1.0  # seconds
MAX_BACKOFF = 30.0  # seconds


def call_openrouter_with_retry(transcript: str) -> dict[str, object] | None:
    """Call OpenRouter API with exponential backoff retry.

    Retries up to MAX_RETRIES times with exponential backoff:
    - Attempt 1: immediate
    - Attempt 2: after 1s
    - Attempt 3: after 2s
    - Attempt 4: after 4s

    Returns structured summary dict or None if all retries exhausted.
    """
    prompt = PROMPT_TEMPLATE.format(transcript=transcript[:2000])

    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES + 1):
        if attempt > 0:
            backoff = min(BASE_BACKOFF * (2 ** (attempt - 1)), MAX_BACKOFF)
            print(f"  Retry {attempt}/{MAX_RETRIES} after {backoff:.1f}s backoff")
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
                    "response_format": {"type": "json_object"},
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            if "choices" not in data or not data["choices"]:
                raise ValueError(f"Unexpected API response: {data}")
            content = data["choices"][0]["message"]["content"]
            return json.loads(content)

        except requests.exceptions.Timeout as e:
            last_error = e
            print(f"  Timeout on attempt {attempt + 1}")
        except requests.exceptions.HTTPError as e:
            last_error = e
            status = e.response.status_code if hasattr(e, "response") else "unknown"
            print(f"  HTTP {status} on attempt {attempt + 1}")
            # Don't retry client errors (4xx) except 429 (rate limit)
            if isinstance(status, int) and status != 429 and 400 <= status < 500:
                return None
        except requests.exceptions.ConnectionError as e:
            last_error = e
            print(f"  Connection error on attempt {attempt + 1}")
        except Exception as e:
            last_error = e
            print(f"  Unexpected error on attempt {attempt + 1}: {e}")

    print(f"  All {MAX_RETRIES + 1} attempts exhausted. Last error: {last_error}")
    return None


def enrich_summaries(db_path: Path = DB_PATH) -> int:
    """Enrich call transcripts with AI summaries."""
    if not OPENROUTER_API_KEY:
        print("  OPENROUTER_API_KEY not set. Skipping enrichment.")
        return 0

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Get interactions without summaries
    rows = cur.execute("""
        SELECT i.id, i.contact_id, i.company_id, i.notes
        FROM interactions i
        LEFT JOIN call_summaries cs ON cs.interaction_id = i.id
        WHERE cs.id IS NULL AND i.notes LIKE 'Transcript: %'
    """).fetchall()

    if not rows:
        print("  No interactions to enrich")
        conn.close()
        return 0

    print(f"  Enriching {len(rows)} call transcripts...")
    enriched = 0

    for row in rows:
        transcript_note = row["notes"].replace("Transcript: ", "")

        # Read actual transcript if available
        transcript_file = (
            Path(__file__).parent.parent
            / "data"
            / "raw"
            / "customer_call_center"
            / "Customer Call Center Dataset + Analysis"
            / "call_transcripts.csv"
        )
        transcript = f"Customer called about {transcript_note}"

        if transcript_file.exists():
            import csv

            with open(transcript_file) as f:
                reader = csv.DictReader(f)
                for t_row in reader:
                    if t_row.get("file_name") == transcript_note:
                        transcript = t_row.get("transcript", transcript)
                        break

        summary = call_openrouter_with_retry(transcript)
        if not summary:
            continue

        summary_id = f"sum_{row['id']}"
        cur.execute(
            """INSERT INTO call_summaries (id, interaction_id, contact_id, company_id,
               transcript, summary, key_topics, objections_raised, sentiment, language,
               action_items, next_steps) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                summary_id,
                row["id"],
                row["contact_id"],
                row["company_id"],
                transcript[:5000],
                summary.get("summary", ""),
                json.dumps(summary.get("key_topics", [])),
                json.dumps(summary.get("objections_raised", [])),
                summary.get("sentiment", "neutral"),
                summary.get("language", "en"),
                json.dumps(summary.get("action_items", [])),
                summary.get("next_steps", ""),
            ),
        )
        enriched += 1
        print(f"  ✓ Enriched: {transcript_note[:50]}...")
        time.sleep(0.5)  # Rate limiting between successful calls

    conn.commit()
    conn.close()
    print(f"  Total enriched: {enriched}")
    return enriched


if __name__ == "__main__":
    enrich_summaries()
