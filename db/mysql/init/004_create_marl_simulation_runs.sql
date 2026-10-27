-- AQQIP MySQL Migration 004
-- Table: marl_simulation_runs
-- Stores MARL scenario results per marl_design_spec.md and data_contracts.md

CREATE TABLE IF NOT EXISTS marl_simulation_runs (
    id                          BIGINT AUTO_INCREMENT PRIMARY KEY,
    simulation_id               VARCHAR(64)     NOT NULL UNIQUE,
    ticker                      VARCHAR(16)     NOT NULL,
    scenario_mode               VARCHAR(32)     NOT NULL,
    scenario_seed               INT             NOT NULL,
    episode_days                INT             NOT NULL,
    initial_price               DECIMAL(18,4)   NOT NULL,
    final_price                 DECIMAL(18,4)   NOT NULL,
    max_drawdown_pct            DECIMAL(8,4)    NOT NULL,
    price_volatility            DECIMAL(8,4)    NULL,
    short_seller_pnl_pct        DECIMAL(8,4)    NOT NULL,
    retail_herd_pnl_pct         DECIMAL(8,4)    NOT NULL,
    market_maker_pnl_pct        DECIMAL(8,4)    NOT NULL,
    market_maker_inventory_peak DECIMAL(12,4)   NULL,
    terminated_reason           VARCHAR(64)     NULL,
    scenario_params_json        JSON            NOT NULL,
    signal_bundle_hash          VARCHAR(64)     NULL,
    reward_config_version       VARCHAR(32)     NULL,
    model_version               VARCHAR(64)     NULL,
    prediction_id               BIGINT          NULL,
    run_at                      TIMESTAMP       DEFAULT CURRENT_TIMESTAMP,
    runtime_ms                  DECIMAL(12,2)   NULL,

    INDEX idx_ticker (ticker),
    INDEX idx_scenario_mode (scenario_mode),
    INDEX idx_run_at (run_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
