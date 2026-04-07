"""Unified SQLite schema for CRM data."""

SCHEMA_SQL = """
-- Companies (merged from accounts.csv + companies_clean)
CREATE TABLE IF NOT EXISTS companies (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    industry TEXT,
    sector TEXT,
    size TEXT,
    year_established INTEGER,
    revenue REAL,
    employees INTEGER,
    location TEXT,
    subsidiary_of TEXT,
    contract_status TEXT,
    payment_behavior TEXT,
    preferred_channel TEXT,
    sales_rep TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Contacts (merged from employees_clean)
CREATE TABLE IF NOT EXISTS contacts (
    id TEXT PRIMARY KEY,
    company_id TEXT NOT NULL,
    name TEXT,
    department TEXT,
    job_title TEXT,
    seniority_level TEXT,
    education_level TEXT,
    work_location TEXT,
    decision_maker INTEGER DEFAULT 0,
    influence_score INTEGER,
    preferred_contact_method TEXT,
    language TEXT,
    last_contact_date TEXT,
    next_followup_date TEXT,
    owner_rep TEXT,
    active INTEGER DEFAULT 1,
    FOREIGN KEY (company_id) REFERENCES companies(id)
);

-- Deals / Opportunities (from sales_pipeline)
CREATE TABLE IF NOT EXISTS deals (
    id TEXT PRIMARY KEY,
    company_id TEXT NOT NULL,
    contact_id TEXT,
    product TEXT NOT NULL,
    deal_stage TEXT NOT NULL,
    engage_date TEXT,
    close_date TEXT,
    close_value REAL,
    sales_agent TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (company_id) REFERENCES companies(id),
    FOREIGN KEY (contact_id) REFERENCES contacts(id)
);

-- Interactions (calls, emails, meetings)
CREATE TABLE IF NOT EXISTS interactions (
    id TEXT PRIMARY KEY,
    contact_id TEXT NOT NULL,
    company_id TEXT NOT NULL,
    interaction_type TEXT NOT NULL,
    direction TEXT,
    duration_seconds INTEGER,
    date TEXT NOT NULL,
    outcome TEXT,
    satisfaction_score REAL,
    notes TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (contact_id) REFERENCES contacts(id),
    FOREIGN KEY (company_id) REFERENCES companies(id)
);

-- Call Summaries (AI-enriched from call_transcripts.csv)
CREATE TABLE IF NOT EXISTS call_summaries (
    id TEXT PRIMARY KEY,
    interaction_id TEXT NOT NULL,
    contact_id TEXT NOT NULL,
    company_id TEXT NOT NULL,
    transcript TEXT,
    summary TEXT,
    key_topics TEXT,
    objections_raised TEXT,
    sentiment TEXT,
    language TEXT,
    action_items TEXT,
    next_steps TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (interaction_id) REFERENCES interactions(id),
    FOREIGN KEY (contact_id) REFERENCES contacts(id),
    FOREIGN KEY (company_id) REFERENCES companies(id)
);

-- Campaigns (from synthetic B2B marketing data)
CREATE TABLE IF NOT EXISTS campaigns (
    id TEXT PRIMARY KEY,
    company_id TEXT NOT NULL,
    campaign_type TEXT,
    marketing_spend REAL,
    leads_generated INTEGER,
    conversion_rate REAL,
    date TEXT,
    FOREIGN KEY (company_id) REFERENCES companies(id)
);

-- Context metadata (for tracking context pieces)
CREATE TABLE IF NOT EXISTS context_pieces (
    id TEXT PRIMARY KEY,
    contact_id TEXT NOT NULL,
    company_id TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_id TEXT NOT NULL,
    content TEXT NOT NULL,
    token_count INTEGER,
    date TEXT NOT NULL,
    metadata TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (contact_id) REFERENCES contacts(id),
    FOREIGN KEY (company_id) REFERENCES companies(id)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_contacts_company ON contacts(company_id);
CREATE INDEX IF NOT EXISTS idx_deals_company ON deals(company_id);
CREATE INDEX IF NOT EXISTS idx_deals_stage ON deals(deal_stage);
CREATE INDEX IF NOT EXISTS idx_interactions_contact ON interactions(contact_id);
CREATE INDEX IF NOT EXISTS idx_interactions_company ON interactions(company_id);
CREATE INDEX IF NOT EXISTS idx_interactions_date ON interactions(date);
CREATE INDEX IF NOT EXISTS idx_call_summaries_contact ON call_summaries(contact_id);
CREATE INDEX IF NOT EXISTS idx_context_pieces_contact ON context_pieces(contact_id);
CREATE INDEX IF NOT EXISTS idx_context_pieces_company ON context_pieces(company_id);
"""
