-- AQQIP MySQL Migration 001
-- Table: ohlcv_prices
-- Stores daily OHLCV market data from yfinance

CREATE TABLE IF NOT EXISTS ohlcv_prices (
    id          BIGINT AUTO_INCREMENT PRIMARY KEY,
    ticker      VARCHAR(16)    NOT NULL,
    trade_date  DATE           NOT NULL,
    open        DECIMAL(18,4)  NOT NULL,
    high        DECIMAL(18,4)  NOT NULL,
    low         DECIMAL(18,4)  NOT NULL,
    close       DECIMAL(18,4)  NOT NULL,
    adj_close   DECIMAL(18,4)  NOT NULL,
    volume      BIGINT         NOT NULL,
    source      VARCHAR(32)    NOT NULL DEFAULT 'yfinance',
    created_at  TIMESTAMP      DEFAULT CURRENT_TIMESTAMP,

    UNIQUE KEY uq_ticker_date (ticker, trade_date),
    INDEX idx_ticker (ticker),
    INDEX idx_trade_date (trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
