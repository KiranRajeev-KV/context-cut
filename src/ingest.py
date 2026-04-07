"""Ingest raw CSV datasets into unified SQLite database."""

import csv
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from schema import SCHEMA_SQL

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
DB_PATH = Path(__file__).parent.parent / "data" / "crm.db"


def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Create database and schema."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


def _uid() -> str:
    return str(uuid.uuid4())


def _safe(val: str | float | None) -> str | None:
    if val is None or (isinstance(val, float) and val != val):  # NaN check
        return None
    return str(val).strip() or None


def _safe_int(val: str | float | None) -> int | None:
    if val is None or (isinstance(val, float) and val != val):
        return None
    try:
        return int(float(str(val).strip()))
    except (ValueError, TypeError):
        return None


def _safe_float(val: str | float | None) -> float | None:
    if val is None or (isinstance(val, float) and val != val):
        return None
    try:
        return float(str(val).strip())
    except (ValueError, TypeError):
        return None


def ingest_companies(conn: sqlite3.Connection) -> dict[str, str]:
    """Ingest companies from accounts.csv and companies_clean.csv.

    Returns mapping of original name/ID → new company ID.
    """
    name_to_id: dict[str, str] = {}
    cur = conn.cursor()

    # From CRM Sales Opportunities: accounts.csv
    accounts_path = RAW_DIR / "crm_sales_opportunities" / "accounts.csv"
    if accounts_path.exists():
        with open(accounts_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = _uid()
                name = _safe(row.get("account"))
                if not name:
                    continue
                name_to_id[name] = cid
                cur.execute(
                    """INSERT INTO companies (id, name, sector, year_established, revenue,
                       employees, location, subsidiary_of) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        cid,
                        name,
                        _safe(row.get("sector")),
                        _safe_int(row.get("year_established")),
                        _safe_float(row.get("revenue")),
                        _safe_int(row.get("employees")),
                        _safe(row.get("office_location")),
                        _safe(row.get("subsidiary_of")),
                    ),
                )

    # From Synthetic B2B: companies_clean.csv
    companies_path = RAW_DIR / "synthetic_b2b_crm_marketing" / "companies_clean_734.csv"
    if companies_path.exists():
        with open(companies_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                company_id_orig = _safe(row.get("Company_ID"))
                if not company_id_orig:
                    continue
                cid = _uid()
                name_to_id[company_id_orig] = cid
                cur.execute(
                    """INSERT INTO companies (id, name, industry, size, revenue,
                       contract_status, payment_behavior, preferred_channel, sales_rep)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        cid,
                        f"Company {company_id_orig}",
                        _safe(row.get("Industry")),
                        _safe(row.get("Company_Size")),
                        _safe_float(row.get("Annual_Revenue (M₺)")),
                        _safe(row.get("Contract_Status")),
                        _safe(row.get("Payment_Behavior")),
                        _safe(row.get("Preferred_Channel")),
                        _safe(row.get("Sales_Rep")),
                    ),
                )

    conn.commit()
    print(f"  Companies ingested: {len(name_to_id)}")
    return name_to_id


def ingest_contacts(
    conn: sqlite3.Connection, company_map: dict[str, str]
) -> dict[str, str]:
    """Ingest contacts from employees_clean.csv.

    Returns mapping of original Employee_ID → new contact ID.
    """
    emp_to_id: dict[str, str] = {}
    cur = conn.cursor()

    employees_path = (
        RAW_DIR / "synthetic_b2b_crm_marketing" / "employees_clean_5234.csv"
    )
    if not employees_path.exists():
        return emp_to_id

    with open(employees_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            emp_id = _safe(row.get("Employee_ID"))
            if not emp_id:
                continue
            company_id_orig = _safe(row.get("Company_ID"))
            company_id = company_map.get(company_id_orig) if company_id_orig else None
            if not company_id:
                continue

            cid = _uid()
            emp_to_id[emp_id] = cid
            cur.execute(
                """INSERT INTO contacts (id, company_id, name, department, job_title,
                   seniority_level, education_level, work_location, decision_maker,
                   influence_score, preferred_contact_method, language,
                   last_contact_date, next_followup_date, owner_rep, active)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    cid,
                    company_id,
                    _safe(row.get("Name")),
                    _safe(row.get("Department")),
                    _safe(row.get("Job_Title")),
                    _safe(row.get("Seniority_Level")),
                    _safe(row.get("Education_Level")),
                    _safe(row.get("Work_Location")),
                    1 if _safe(row.get("Decision_Maker_Flag")) == "Yes" else 0,
                    _safe_int(row.get("Influence_Score")),
                    _safe(row.get("Preferred_Contact_Method")),
                    _safe(row.get("Language")),
                    _safe(row.get("Last_Contact_Date")),
                    _safe(row.get("Next_Followup_Date")),
                    _safe(row.get("Owner_Rep")),
                    1 if _safe(row.get("Active_Flag")) == "Yes" else 0,
                ),
            )

    conn.commit()
    print(f"  Contacts ingested: {len(emp_to_id)}")
    return emp_to_id


def ingest_deals(conn: sqlite3.Connection, company_map: dict[str, str]) -> int:
    """Ingest deals from sales_pipeline.csv."""
    cur = conn.cursor()
    count = 0

    deals_path = RAW_DIR / "crm_sales_opportunities" / "sales_pipeline.csv"
    if not deals_path.exists():
        return count

    with open(deals_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            account = _safe(row.get("account"))
            company_id = company_map.get(account) if account else None
            if not company_id:
                continue

            cur.execute(
                """INSERT INTO deals (id, company_id, product, deal_stage, engage_date,
                   close_date, close_value, sales_agent) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    _uid(),
                    company_id,
                    _safe(row.get("product")),
                    _safe(row.get("deal_stage")),
                    _safe(row.get("engage_date")),
                    _safe(row.get("close_date")),
                    _safe_float(row.get("close_value")),
                    _safe(row.get("sales_agent")),
                ),
            )
            count += 1

    conn.commit()
    print(f"  Deals ingested: {count}")
    return count


def ingest_interactions(conn: sqlite3.Connection, company_map: dict[str, str]) -> int:
    """Ingest call interactions from call_transcripts.csv."""
    cur = conn.cursor()
    count = 0

    transcripts_path = (
        RAW_DIR
        / "customer_call_center"
        / "Customer Call Center Dataset + Analysis"
        / "call_transcripts.csv"
    )
    if not transcripts_path.exists():
        return count

    with open(transcripts_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_name = _safe(row.get("file_name"))
            if not file_name:
                continue

            # Assign to a random company/contact for demo purposes
            # (real data would have proper linkage)
            company_id = cur.execute(
                "SELECT id FROM companies ORDER BY RANDOM() LIMIT 1"
            ).fetchone()
            if not company_id:
                continue
            company_id = company_id[0]

            contact_id = cur.execute(
                "SELECT id FROM contacts WHERE company_id = ? ORDER BY RANDOM() LIMIT 1",
                (company_id,),
            ).fetchone()
            if not contact_id:
                continue
            contact_id = contact_id[0]

            iid = _uid()
            date = datetime.now() - timedelta(days=count * 3)
            cur.execute(
                """INSERT INTO interactions (id, contact_id, company_id, interaction_type,
                   direction, date, notes) VALUES (?, ?, ?, 'call', 'inbound', ?, ?)""",
                (
                    iid,
                    contact_id,
                    company_id,
                    date.isoformat(),
                    f"Transcript: {file_name}",
                ),
            )
            count += 1

    conn.commit()
    print(f"  Interactions ingested: {count}")
    return count


def ingest_all(db_path: Path = DB_PATH) -> None:
    """Run full ingestion pipeline."""
    print("Initializing database...")
    conn = init_db(db_path)

    print("\nIngesting companies...")
    company_map = ingest_companies(conn)

    print("\nIngesting contacts...")
    ingest_contacts(conn, company_map)

    print("\nIngesting deals...")
    ingest_deals(conn, company_map)

    print("\nIngesting interactions...")
    ingest_interactions(conn, company_map)

    # Summary
    cur = conn.cursor()
    for table in [
        "companies",
        "contacts",
        "deals",
        "interactions",
        "call_summaries",
        "campaigns",
    ]:
        row = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        print(f"  {table}: {row[0]} rows")

    conn.close()
    print(f"\nDatabase saved to: {db_path}")


if __name__ == "__main__":
    ingest_all()
