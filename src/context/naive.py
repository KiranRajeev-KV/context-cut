"""Naive context assembler — dumps ALL context into the prompt.

This is the baseline approach that the CTO's paper identifies as the problem:
"thoughtless context dumping" causes O(n²) attention cost, context poisoning,
and unpredictable cache-miss latency spikes.
"""

import sqlite3
from pathlib import Path

import tiktoken

DB_PATH = Path(__file__).parent.parent.parent / "data" / "crm.db"


def _get_conn(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def assemble_naive_context(company_name: str, contact_name: str) -> str:
    """Concatenate ALL available CRM context for a contact/company.

    This is the naive append approach from the paper:
    Q1 = C1 + C2 + C3 → complexity ∝ n(Q1)²

    Returns the full context string and metadata about token count.
    """
    conn = _get_conn()

    # Get company
    company = conn.execute(
        "SELECT * FROM companies WHERE name LIKE ? LIMIT 1",
        (f"%{company_name}%",),
    ).fetchone()

    if not company:
        conn.close()
        return f"Company '{company_name}' not found."

    company_id = company["id"]

    # Get contact
    contact = conn.execute(
        "SELECT * FROM contacts WHERE name LIKE ? AND company_id = ? LIMIT 1",
        (f"%{contact_name}%", company_id),
    ).fetchone()

    if not contact:
        conn.close()
        return f"Contact '{contact_name}' not found at {company['name']}."

    contact_id = contact["id"]

    # Get all deals
    deals = conn.execute(
        "SELECT * FROM deals WHERE company_id = ? ORDER BY engage_date DESC",
        (company_id,),
    ).fetchall()

    # Get all call summaries
    calls = conn.execute(
        """SELECT cs.*, i.date, i.duration_seconds, i.direction
           FROM call_summaries cs
           JOIN interactions i ON i.id = cs.interaction_id
           WHERE cs.contact_id = ? OR cs.company_id = ?
           ORDER BY i.date ASC""",
        (contact_id, company_id),
    ).fetchall()

    # Get all interactions
    interactions = conn.execute(
        """SELECT * FROM interactions
           WHERE contact_id = ? OR company_id = ?
           ORDER BY date ASC""",
        (contact_id, company_id),
    ).fetchall()

    conn.close()

    # Build the massive context string — dump everything
    context_parts = []

    # Company info
    context_parts.append("## COMPANY INFORMATION")
    context_parts.append(f"Name: {company['name']}")
    context_parts.append(f"Sector: {company['sector']}")
    context_parts.append(f"Industry: {company['industry']}")
    context_parts.append(f"Size: {company['size']}")
    context_parts.append(f"Revenue: {company['revenue']}")
    context_parts.append(f"Employees: {company['employees']}")
    context_parts.append(f"Location: {company['location']}")
    context_parts.append(f"Contract Status: {company['contract_status']}")
    context_parts.append(f"Payment Behavior: {company['payment_behavior']}")
    context_parts.append(f"Preferred Channel: {company['preferred_channel']}")
    context_parts.append("")

    # Contact info
    context_parts.append("## CONTACT INFORMATION")
    context_parts.append(f"Name: {contact['name']}")
    context_parts.append(f"Job Title: {contact['job_title']}")
    context_parts.append(f"Department: {contact['department']}")
    context_parts.append(f"Seniority: {contact['seniority_level']}")
    context_parts.append(f"Decision Maker: {bool(contact['decision_maker'])}")
    context_parts.append(f"Influence Score: {contact['influence_score']}")
    context_parts.append(f"Preferred Contact: {contact['preferred_contact_method']}")
    context_parts.append(f"Language: {contact['language']}")
    context_parts.append(f"Last Contact: {contact['last_contact_date']}")
    context_parts.append(f"Next Follow-up: {contact['next_followup_date']}")
    context_parts.append("")

    # All deals (even irrelevant ones)
    context_parts.append(f"## ALL DEALS ({len(deals)} total)")
    for deal in deals:
        context_parts.append(
            f"- {deal['product']} | Stage: {deal['deal_stage']} | "
            f"Value: {deal['close_value']} | Engaged: {deal['engage_date']} | "
            f"Closed: {deal['close_date']} | Agent: {deal['sales_agent']}"
        )
    context_parts.append("")

    # All call summaries (full transcripts + summaries)
    context_parts.append(f"## CALL HISTORY ({len(calls)} calls)")
    for call in calls:
        context_parts.append(
            f"### Call on {call['date']} ({call['duration_seconds']}s, {call['direction']})"
        )
        if call["transcript"]:
            context_parts.append(f"Transcript: {call['transcript']}")
        if call["summary"]:
            context_parts.append(f"Summary: {call['summary']}")
        if call["key_topics"]:
            context_parts.append(f"Topics: {call['key_topics']}")
        if call["objections_raised"]:
            context_parts.append(f"Objections: {call['objections_raised']}")
        if call["sentiment"]:
            context_parts.append(f"Sentiment: {call['sentiment']}")
        if call["action_items"]:
            context_parts.append(f"Action Items: {call['action_items']}")
        if call["next_steps"]:
            context_parts.append(f"Next Steps: {call['next_steps']}")
        context_parts.append("")

    # All interactions
    context_parts.append(f"## ALL INTERACTIONS ({len(interactions)} total)")
    for interaction in interactions:
        context_parts.append(
            f"- {interaction['date']} | {interaction['interaction_type']} | "
            f"{interaction['direction']} | {interaction['duration_seconds']}s | "
            f"Outcome: {interaction['outcome']}"
        )
    context_parts.append("")

    full_context = "\n".join(context_parts)

    return full_context


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken (cl100k_base for GPT-4 / most modern models)."""
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))
