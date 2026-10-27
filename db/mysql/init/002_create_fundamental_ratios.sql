-- AQQIP MySQL Migration 002
-- Table: fundamental_ratios
-- Stores computed Beneish M-Score, Altman Z-Score, and raw feature components

CREATE TABLE IF NOT EXISTS fundamental_ratios (
    id                  BIGINT AUTO_INCREMENT PRIMARY KEY,
    ticker              VARCHAR(16)    NOT NULL,
    cik                 VARCHAR(10)    NULL,
    fiscal_period       VARCHAR(16)    NOT NULL,
    filing_date         DATE           NULL,
    beneish_m_score     DECIMAL(10,4)  NULL,
    altman_z_score      DECIMAL(10,4)  NULL,
    raw_features_json   JSON           NULL,
    source_filing_accession VARCHAR(30) NULL,
    feature_completeness DECIMAL(5,4)  NULL,
    computed_at         TIMESTAMP      DEFAULT CURRENT_TIMESTAMP,

    UNIQUE KEY uq_ticker_period (ticker, fiscal_period),
    INDEX idx_ticker (ticker),
    INDEX idx_ticker_filing_date (ticker, filing_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
