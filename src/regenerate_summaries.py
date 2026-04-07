"""Regenerate call summaries with deal-stage-appropriate content.

Fixes the issue where all calls had identical "Initial outreach" text
regardless of deal stage (Won, Lost, Prospecting, Engaging).
"""

import json
import random
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "crm.db"

# Stage-appropriate summary templates
STAGE_TEMPLATES = {
    "Won": [
        "Deal closed successfully. Customer confirmed {product} purchase at ${value}. Onboarding scheduled for next week. Customer expressed satisfaction with pricing and implementation timeline. All contract terms finalized.",
        "Closed won. {product} deal confirmed at ${value}. Customer approved final proposal. Implementation team notified. Customer requested expedited onboarding. Payment terms: Net 30.",
        "Successfully closed {product} deal. Customer signed contract at ${value}. Positive feedback throughout sales process. Next steps: kickoff meeting, account handoff to customer success team.",
    ],
    "Lost": [
        "Deal lost to competitor. Customer chose {product} alternative due to lower pricing. Budget constraints cited as primary factor. Customer open to re-engagement in 6-12 months. Added to nurture campaign.",
        "Lost deal. Customer selected competitor offering with better feature fit for their {sector} use case. Price was secondary concern. Requested to be contacted when roadmap includes requested features.",
        "Deal closed lost. Customer went with competitor after final comparison. Main objections: implementation timeline too long, preferred vendor already in their stack. Professional relationship maintained.",
    ],
    "Engaging": [
        "Qualification call. Discussed current pain points in {sector} operations. Customer interested in {product} but needs internal stakeholder alignment. Requested case studies and ROI calculator. Follow-up scheduled.",
        "Discovery call completed. Customer evaluating options for {product}. Key requirements: integration with existing stack, budget approval needed from finance team. Sent technical documentation and pricing tiers.",
        "Engagement call. Customer actively comparing vendors. Discussed {product} capabilities vs competitors. Customer requested product demo for technical team. Decision timeline: 4-6 weeks.",
    ],
    "Prospecting": [
        "Initial outreach. Introduced {product} platform to {sector} company. Customer expressed mild interest but not actively evaluating. Sent overview materials. Will follow up in 2-3 weeks.",
        "First contact call. Briefly discussed {product} value proposition. Customer receptive but timing not right. Agreed to reconnect next quarter. Added to targeted outreach list.",
        "Prospecting call. Identified potential fit for {product}. Customer currently satisfied with existing solution but open to future evaluation. Established rapport and sent introductory materials.",
    ],
}

OBJECTIONS_BY_STAGE = {
    "Won": [],
    "Lost": [
        "budget constraints",
        "competitor had better pricing",
        "implementation timeline too long",
        "preferred existing vendor",
        "missing specific feature",
    ],
    "Engaging": [
        "needs internal approval",
        "evaluating multiple vendors",
        "budget not finalized",
        "concerned about migration effort",
    ],
    "Prospecting": [
        "not actively looking",
        "satisfied with current solution",
        "timing not right",
        "needs to understand ROI first",
    ],
}

ACTIONS_BY_STAGE = {
    "Won": [
        "begin onboarding",
        "schedule kickoff meeting",
        "hand off to customer success",
        "send welcome package",
    ],
    "Lost": [
        "add to nurture campaign",
        "document loss reasons",
        "schedule 6-month check-in",
        "notify product team of feature gap",
    ],
    "Engaging": [
        "send case studies",
        "schedule product demo",
        "connect with technical team",
        "send ROI calculator",
    ],
    "Prospecting": [
        "send product overview",
        "follow up in 3 weeks",
        "add to targeted outreach",
        "share relevant industry report",
    ],
}


def regenerate_summaries(db_path: Path = DB_PATH) -> int:
    """Regenerate all synthetic call summaries with stage-appropriate content."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Get all synthetic call summaries with deal context
    # Use subquery to pick ONE deal per company (most recent) to avoid cross-product
    rows = conn.execute("""
        SELECT cs.id, cs.contact_id, cs.company_id,
               ct.name as contact_name, ct.job_title,
               c.name as company_name, c.sector,
               d.deal_stage, d.product, d.close_value,
               i.date
        FROM call_summaries cs
        JOIN contacts ct ON ct.id = cs.contact_id
        JOIN companies c ON c.id = cs.company_id
        JOIN interactions i ON i.id = cs.interaction_id
        JOIN deals d ON d.company_id = cs.company_id
            AND d.id = (
                SELECT d2.id FROM deals d2
                WHERE d2.company_id = cs.company_id
                ORDER BY d2.engage_date DESC
                LIMIT 1
            )
        WHERE cs.id LIKE 'syn_%'
    """).fetchall()

    if not rows:
        print("  No synthetic summaries to regenerate")
        conn.close()
        return 0

    print(f"  Regenerating {len(rows)} synthetic summaries...")
    updated = 0

    for row in rows:
        stage = row["deal_stage"]
        templates = STAGE_TEMPLATES.get(stage, STAGE_TEMPLATES["Prospecting"])
        template = random.choice(templates)

        summary = template.format(
            product=row["product"] or "our solution",
            value=row["close_value"] or "custom",
            sector=row["sector"] or "their",
        )

        objections = random.sample(
            OBJECTIONS_BY_STAGE.get(stage, []),
            min(2, len(OBJECTIONS_BY_STAGE.get(stage, []))),
        )
        actions = random.sample(
            ACTIONS_BY_STAGE.get(stage, []),
            min(2, len(ACTIONS_BY_STAGE.get(stage, []))),
        )

        sentiment = (
            "positive" if stage == "Won" else random.choice(["neutral", "negative"])
        )

        conn.execute(
            """UPDATE call_summaries SET
               summary = ?, key_topics = ?, objections_raised = ?,
               action_items = ?, next_steps = ?, sentiment = ?
               WHERE id = ?""",
            (
                summary,
                json.dumps([stage.lower(), row["product"] or ""]),
                json.dumps(objections),
                json.dumps(actions),
                "Follow up within 3 business days",
                sentiment,
                row["id"],
            ),
        )
        updated += 1

    conn.commit()
    conn.close()
    print(f"  Regenerated {updated} summaries")

    # Verify variety
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    for stage in ["Won", "Lost", "Engaging", "Prospecting"]:
        sample = conn.execute(
            """
            SELECT cs.summary, cs.sentiment
            FROM call_summaries cs
            JOIN deals d ON d.company_id = cs.company_id
            WHERE cs.id LIKE 'syn_%' AND d.deal_stage = ?
            LIMIT 1
        """,
            (stage,),
        ).fetchone()
        if sample:
            print(f"  [{stage}] {sample['sentiment']}: {sample['summary'][:80]}...")
    conn.close()

    return updated


if __name__ == "__main__":
    regenerate_summaries()
