-- AQQIP MySQL Migration 003
-- Table: predictions
-- Stores model predictions with full lifecycle tracking per mlops_contract.md

CREATE TABLE IF NOT EXISTS predictions (
    id                          BIGINT AUTO_INCREMENT PRIMARY KEY,
    ticker                      VARCHAR(16)     NOT NULL,
    cik                         VARCHAR(10)     NULL,
    model_run_id                VARCHAR(64)     NOT NULL,
    model_version               VARCHAR(64)     NULL,
    predicted_drop_probability  DECIMAL(6,5)    NOT NULL,
    calibrated_probability      DECIMAL(6,5)    NULL,
    input_feature_hash          VARCHAR(64)     NULL,
    predicted_at                TIMESTAMP       NOT NULL,
    target_date                 DATE            NOT NULL,
    prediction_horizon_days     INT             NOT NULL DEFAULT 20,
    drop_threshold_pct          DECIMAL(5,2)    NOT NULL DEFAULT 10.00,
    actual_drop_observed        BOOLEAN         NULL,
    actual_price_change_pct     DECIMAL(8,4)    NULL,
    resolved_at                 TIMESTAMP       NULL,
    status                      ENUM(
                                    'generated',
                                    'open',
                                    'eligible',
                                    'resolving',
                                    'resolved',
                                    'resolution_failed',
                                    'data_unavailable',
                                    'invalid_target'
                                ) NOT NULL DEFAULT 'open',

    INDEX idx_ticker (ticker),
    INDEX idx_status (status),
    INDEX idx_target_date (target_date),
    INDEX idx_model_run (model_run_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
