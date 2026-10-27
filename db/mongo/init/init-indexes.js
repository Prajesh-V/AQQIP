// AQQIP MongoDB Initialization
// Creates collections with required indexes per architecture.md and data_contracts.md

db = db.getSiblingDB('aqqip_raw');

// --- sec_filings collection ---
db.createCollection('sec_filings');

// Accession number is the canonical filing identity — must be unique
db.sec_filings.createIndex(
    { "accession_number": 1 },
    { unique: true, name: "uq_accession_number" }
);

// Chronological retrieval: find all filings for a ticker sorted by fiscal year
db.sec_filings.createIndex(
    { "ticker": 1, "fiscal_year": 1, "filing_type": 1 },
    { name: "idx_ticker_year_type" }
);

// Processing status for pipeline state queries
db.sec_filings.createIndex(
    { "processing_status": 1 },
    { name: "idx_processing_status" }
);

// CIK lookup
db.sec_filings.createIndex(
    { "cik": 1 },
    { name: "idx_cik" }
);

// --- semantic_diffs collection ---
db.createCollection('semantic_diffs');

// Primary lookup: find diffs for a ticker/section/year comparison
db.semantic_diffs.createIndex(
    { "ticker": 1, "section": 1, "year_from": 1, "year_to": 1 },
    { unique: true, name: "uq_ticker_section_years" }
);

// Chronological ordering
db.semantic_diffs.createIndex(
    { "computed_at": -1 },
    { name: "idx_computed_at" }
);

// --- ticker_universe collection ---
// Stores tracked tickers with CIK resolution and freshness state
db.createCollection('ticker_universe');

db.ticker_universe.createIndex(
    { "ticker": 1 },
    { unique: true, name: "uq_ticker" }
);

db.ticker_universe.createIndex(
    { "cik": 1 },
    { name: "idx_cik" }
);

print("AQQIP MongoDB initialization complete.");
