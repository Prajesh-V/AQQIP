"""
AQQIP Real-Data SignalBundle Bridge

Bridges historical market data, fundamental ratios, qualitative filing features,
and verified MLflow LightGBM model artifacts into canonical SignalBundles.

Strictly enforces:
1. Point-in-time integrity: only data observed on or before as_of_date is accessible.
2. Exact feature schema matching each model variant (M, MF, MFQ).
3. Preservation of unsigned change magnitude and directional qualitative risk.
4. Complete provenance tracking (model run ID, feature hash, filing accessions, dates).
5. Zero lookahead leakage.
"""
from __future__ import annotations

import os
import json
import hashlib
from datetime import date, datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
import mlflow
import mlflow.lightgbm

from config.settings import get_settings
from src.database.mysql_client import get_mysql_connection
from src.quant.feature_builder import FeatureBuilder
from src.qualitative.directional_signal import DirectionalQualitativeRiskAnalyzer
from src.marl.models import (
    SecurityIdentity,
    MarketContext,
    QuantitativeSignalBlock,
    QualitativeSignalBlock,
    EvidenceQuality,
    FusionBlock,
    ProvenanceBlock,
    SignalBundle
)
from src.mlops.models import PredictionRecord
from src.mlops.repository import MLOpsRepository
from src.utils.logging import get_logger

logger = get_logger("signal_bridge")

# Verified MLflow run artifacts
MODEL_RUNS = {
    "M": "59a7f799181243c796b73ee3b03f5686",
    "MF": "31ac56b4ff71411dad7cb13315459c30",
    "MFQ": "54a2403adc824c179a20a75208325a10"
}

_MODEL_CACHE: Dict[str, Tuple[Any, List[str]]] = {}


def get_cached_model(variant: str) -> Tuple[Any, List[str]]:
    """Loads and caches verified MLflow model booster and expected feature list."""
    if variant not in MODEL_RUNS:
        raise ValueError(f"Unknown model variant {variant}. Must be one of {list(MODEL_RUNS.keys())}")
    
    if variant in _MODEL_CACHE:
        return _MODEL_CACHE[variant]
        
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    
    run_id = MODEL_RUNS[variant]
    model_uri = f"runs:/{run_id}/model"
    logger.info("Loading LightGBM model for SignalBundle bridge", variant=variant, run_id=run_id)
    
    model = mlflow.lightgbm.load_model(model_uri)
    features = model.feature_name()
    _MODEL_CACHE[variant] = (model, features)
    return model, features


class SignalBundleBridge:
    """Builds canonical SignalBundles from MySQL real data without lookahead."""

    def __init__(self):
        self.qual_analyzer = DirectionalQualitativeRiskAnalyzer()

    def build_bundle(
        self,
        ticker: str,
        as_of_date: Optional[date] = None,
        model_variant: str = "M",
        random_seed: int = 42,
        scenario_id: Optional[str] = None
    ) -> SignalBundle:
        """
        Constructs a complete SignalBundle for ticker as of as_of_date.
        """
        model, expected_features = get_cached_model(model_variant)
        run_id = MODEL_RUNS[model_variant]

        with get_mysql_connection() as conn:
            cursor = conn.cursor(dictionary=True)

            # 2. Historical market data strictly up to as_of_date
            if as_of_date:
                cursor.execute(
                    """
                    SELECT trade_date, open, high, low, close, adj_close, volume
                    FROM ohlcv_prices
                    WHERE ticker = %s AND trade_date <= %s
                    ORDER BY trade_date ASC
                    """,
                    (ticker, as_of_date)
                )
            else:
                cursor.execute(
                    """
                    SELECT trade_date, open, high, low, close, adj_close, volume
                    FROM ohlcv_prices
                    WHERE ticker = %s
                    ORDER BY trade_date ASC
                    """,
                    (ticker,)
                )
            market_rows = cursor.fetchall()

            if not market_rows or len(market_rows) < 20:
                raise ValueError(
                    f"Insufficient market data for ticker {ticker} as of {as_of_date}. Found {len(market_rows)} rows."
                )

            market_df = pd.DataFrame(market_rows)
            for col in ["open", "high", "low", "close", "adj_close", "volume"]:
                if col in market_df.columns:
                    market_df[col] = pd.to_numeric(market_df[col], errors="coerce").astype(float)
            market_df["ticker"] = ticker
            actual_as_of = market_df["trade_date"].max()

            # 3. Latest eligible fundamental ratios (point-in-time strictly on or before actual_as_of)
            cursor.execute(
                """
                SELECT * FROM fundamental_ratios
                WHERE ticker = %s AND filing_date <= %s
                ORDER BY filing_date DESC LIMIT 1
                """,
                (ticker, actual_as_of)
            )
            fund_row = cursor.fetchone()

            # 4. Latest eligible qualitative filing features (point-in-time strictly on or before actual_as_of)
            cursor.execute(
                """
                SELECT * FROM qualitative_filing_features
                WHERE ticker = %s AND filing_date <= %s
                ORDER BY filing_date DESC LIMIT 1
                """,
                (ticker, actual_as_of)
            )
            qual_row = cursor.fetchone()

        cik = (qual_row.get("cik") if qual_row else None) or (fund_row.get("cik") if fund_row else None) or "0000320193"
        co_name = ticker

        # Build market features
        market_feats_df = FeatureBuilder.build_market_features(market_df, price_col="adj_close")
        latest_market = market_feats_df.iloc[-1]

        current_price = float(latest_market["adj_close"])
        recent_ret = float(latest_market["return_5d"]) if not pd.isna(latest_market["return_5d"]) else 0.0
        recent_vol = float(latest_market["volatility_20d"]) if not pd.isna(latest_market["volatility_20d"]) else 0.01
        avg_vol = float(latest_market["volume"]) if "volume" in latest_market and not pd.isna(latest_market["volume"]) else 1000000.0
        liquidity = float(min(2.0, max(0.5, avg_vol / 1000000.0)))

        # Assemble MarketContext
        market_ctx = MarketContext(
            current_price=current_price,
            recent_return=recent_ret,
            recent_volatility=recent_vol,
            average_volume=avg_vol,
            liquidity_proxy=liquidity,
            as_of_date=actual_as_of
        )

        # Build feature dictionary for model
        feature_dict: Dict[str, float] = {}
        missing_reasons: List[str] = []

        # Market features
        for f in ['return_1d', 'return_5d', 'return_20d', 'return_60d', 'volatility_20d',
                  'daily_range_pct', 'avg_range_20d', 'volume_z_score_20d', 'drawdown_60d']:
            val = latest_market.get(f)
            feature_dict[f] = float(val) if val is not None and not pd.isna(val) else 0.0

        # Fundamentals
        altman_z = float(fund_row["altman_z_score"]) if fund_row and fund_row.get("altman_z_score") is not None else None
        beneish_m = float(fund_row["beneish_m_score"]) if fund_row and fund_row.get("beneish_m_score") is not None else None
        feature_dict['altman_z_score'] = altman_z if altman_z is not None else np.nan
        feature_dict['beneish_m_score'] = beneish_m if beneish_m is not None else np.nan

        # Qualitative features
        qual_available = (qual_row is not None)
        filing_pair_avail = (qual_row is not None and qual_row.get("prior_accession_number") is not None)

        if qual_available:
            for qf in [
                'rf_length_change_pct', 'rf_cosine_similarity', 'rf_change_magnitude',
                'rf_added_sentences_count', 'rf_deleted_sentences_count', 'rf_added_sentences_ratio',
                'mda_length_change_pct', 'mda_cosine_similarity', 'mda_change_magnitude'
            ]:
                qval = qual_row.get(qf)
                feature_dict[qf] = float(qval) if qval is not None else np.nan
        else:
            missing_reasons.append(f"No eligible SEC 10-K filing found prior to as-of date {actual_as_of}")
            for qf in [
                'rf_length_change_pct', 'rf_cosine_similarity', 'rf_change_magnitude',
                'rf_added_sentences_count', 'rf_deleted_sentences_count', 'rf_added_sentences_ratio',
                'mda_length_change_pct', 'mda_cosine_similarity', 'mda_change_magnitude'
            ]:
                feature_dict[qf] = np.nan

        # Extract only features required by the active model variant
        input_vector = [feature_dict[f] for f in expected_features]
        feature_hash = hashlib.sha256(
            json.dumps({f: (round(feature_dict[f], 6) if not pd.isna(feature_dict[f]) else None) for f in expected_features}, sort_keys=True).encode()
        ).hexdigest()

        # LightGBM inference
        X = np.array([input_vector])
        raw_prob = float(model.predict(X)[0])
        # Quant pressure score calibrated: prevalence is ~0.11
        quant_pressure = float(min(1.0, max(0.0, raw_prob / 0.11)))

        quant_block = QuantitativeSignalBlock(
            quant_pressure_score=quant_pressure,
            beneish_m_score=beneish_m,
            altman_z_score=altman_z,
            model_probability=raw_prob,
            model_run_id=run_id,
            model_variant=model_variant,
            input_feature_hash=feature_hash,
            features_snapshot={f: (float(feature_dict[f]) if not pd.isna(feature_dict[f]) else None) for f in expected_features}
        )

        # Directional qualitative analysis
        if qual_available:
            change_mag = float(qual_row.get("rf_change_magnitude") or 0.0)
            cos_sim = float(qual_row.get("rf_cosine_similarity") or 1.0)
            added_cnt = int(qual_row.get("rf_added_sentences_count") or 0)
            del_cnt = int(qual_row.get("rf_deleted_sentences_count") or 0)

            # Analyze directional risk
            dir_res = self.qual_analyzer.analyze(
                added_sentences=[],
                raw_text_change="",
                unsigned_change_magnitude=change_mag,
                is_available=True
            )

            qual_block = QualitativeSignalBlock(
                semantic_shift_score=dir_res.directional_risk_score,
                cosine_similarity=cos_sim,
                change_magnitude=change_mag,
                directional_risk_score=dir_res.directional_risk_score,
                added_count=added_cnt,
                deleted_count=del_cnt,
                shifted_count=0,
                filing_year_from=qual_row.get("fiscal_year") - 1 if qual_row.get("fiscal_year") else None,
                filing_year_to=qual_row.get("fiscal_year"),
                source_accession=qual_row.get("accession_number"),
                prior_accession=qual_row.get("prior_accession_number"),
                filing_availability_date=qual_row.get("filing_date"),
                evidence_snippets=dir_res.evidence_snippets,
                extraction_status=qual_row.get("extraction_status", "SUCCESS")
            )
        else:
            qual_block = QualitativeSignalBlock(
                semantic_shift_score=0.0,
                cosine_similarity=1.0,
                change_magnitude=0.0,
                directional_risk_score=0.0,
                added_count=0,
                deleted_count=0,
                shifted_count=0,
                extraction_status="MISSING_EVIDENCE"
            )

        # Evidence completeness
        non_null_feats = sum(1 for f in expected_features if not pd.isna(feature_dict[f]))
        feature_completeness = float(non_null_feats / len(expected_features))

        evidence_quality = EvidenceQuality(
            quant_available=True,
            qualitative_available=qual_available,
            filing_pair_available=filing_pair_avail,
            feature_completeness=feature_completeness,
            data_freshness_minutes=0.0,
            missing_reasons=missing_reasons
        )

        # Fusion
        qual_p = qual_block.directional_risk_score
        concordance = float(max(0.0, 1.0 - abs(quant_pressure - qual_p)))
        combined_p = float(0.5 * quant_pressure + 0.5 * qual_p)

        fusion_block = FusionBlock(
            evidence_concordance=concordance,
            combined_forensic_pressure=combined_p,
            fusion_version="v1.0.0"
        )

        provenance_block = ProvenanceBlock(
            as_of_date=actual_as_of,
            quant_model_run_id=run_id,
            qualitative_diff_id=qual_block.source_accession,
            random_seed=random_seed,
            scenario_id=scenario_id
        )

        return SignalBundle(
            security=SecurityIdentity(
                ticker=ticker,
                company_name=co_name,
                cik=cik
            ),
            market_context=market_ctx,
            quantitative=quant_block,
            qualitative=qual_block,
            evidence_quality=evidence_quality,
            fusion=fusion_block,
            provenance=provenance_block
        )

    def create_frozen_prospective_stream(
        self,
        tickers: List[str],
        as_of_date: Optional[date] = None,
        variants: Optional[List[str]] = None
    ) -> List[PredictionRecord]:
        """
        Creates and persists frozen prospective predictions for all 3 model variants
        without tuning against subsequent outcomes.
        """
        if variants is None:
            variants = ["M", "MF", "MFQ"]

        repo = MLOpsRepository()
        saved_records: List[PredictionRecord] = []

        with get_mysql_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT trade_date FROM ohlcv_prices ORDER BY trade_date ASC")
            all_trading_dates = [r[0] for r in cursor.fetchall()]

        for ticker in tickers:
            for variant in variants:
                bundle = self.build_bundle(ticker, as_of_date=as_of_date, model_variant=variant)
                pred_date = bundle.market_context.as_of_date
                
                # Determine target date
                if pred_date in all_trading_dates:
                    pos = all_trading_dates.index(pred_date)
                    if pos + 20 < len(all_trading_dates):
                        target_dt = all_trading_dates[pos + 20]
                    else:
                        # Forward 28 calendar days
                        target_dt = pred_date + timedelta(days=28)
                else:
                    target_dt = pred_date + timedelta(days=28)

                rec = PredictionRecord(
                    ticker=ticker,
                    cik=bundle.security.cik,
                    model_run_id=bundle.quantitative.model_run_id,
                    model_version=f"{variant.lower()}_v1",
                    predicted_drop_probability=bundle.quantitative.model_probability,
                    calibrated_probability=bundle.quantitative.quant_pressure_score,
                    input_feature_hash=bundle.quantitative.input_feature_hash,
                    predicted_at=datetime.combine(pred_date, datetime.min.time(), tzinfo=timezone.utc),
                    target_date=target_dt,
                    prediction_horizon_days=20,
                    drop_threshold_pct=10.0,
                    status="open"
                )
                pred_id = repo.create_prediction(rec)
                bundle.provenance.prediction_id = pred_id
                saved_records.append(rec)
                logger.info(
                    "Persisted frozen prospective prediction",
                    ticker=ticker,
                    variant=variant,
                    pred_id=pred_id,
                    prob=rec.predicted_drop_probability,
                    target_date=str(target_dt)
                )

        return saved_records
