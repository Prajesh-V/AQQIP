"""
AQQIP Chronological Rolling-Origin Model Evaluator

Implements multi-fold expanding-window temporal validation on pre-2025 development data:
- Folds: 2021, 2022, 2023, 2024 out-of-time test folds.
- Strict boundary purging: 20 trading sessions purged before each validation start.
- Quantile threshold (alpha=0.10) fitted exclusively on each training fold.
- Preprocessing, scaling, and calibration fitted exclusively on each training fold.
- Paired comparison across identical rows for:
  1. Market-Only LightGBM (M)
  2. Regularized Logistic Regression (LogReg)
  3. Carefully Regularized LightGBM with Cross-Sectional Features (M_Regularized)
  4. Quant-Integrated (MF)
  5. Qual-Quant Integrated (MFQ)
- Reports out-of-time ROC-AUC, PR-AUC, Brier score, and Brier skill score (deployable vs train prevalence).
"""
from __future__ import annotations

import os
import sys
from datetime import date, datetime
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV

from config.settings import get_settings
from src.quant.train_pipeline import (
    load_raw_data,
    DEFAULT_MARKET_FEATURES,
    DEFAULT_FUNDAMENTAL_FEATURES,
    DEFAULT_QUALITATIVE_FEATURES
)
from src.quant.point_in_time_join import PointInTimeJoiner
from src.quant.feature_builder import FeatureBuilder
from src.quant.target_builder import TargetBuilder
from src.quant.benchmark_runner import compute_calibration_diagnostics, select_operational_threshold
from src.utils.logging import get_logger

logger = get_logger("rolling_origin_evaluator")


def build_enhanced_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes standard market features plus robust cross-sectional relative features:
    - Relative return: return_20d - cross_sectional_mean(return_20d)
    - Relative volatility: volatility_20d / cross_sectional_mean(volatility_20d)
    - Market breadth: fraction of universe with positive 20d return
    All computed strictly cross-sectionally per trade_date (zero lookahead).
    """
    panel = FeatureBuilder.build_market_features(df, price_col="adj_close")
    
    # Cross-sectional relative features per trade_date
    cs_group = panel.groupby("trade_date")
    panel["cs_mean_ret20d"] = cs_group["return_20d"].transform("mean")
    panel["return_20d_rel"] = panel["return_20d"] - panel["cs_mean_ret20d"]
    
    panel["cs_mean_vol20d"] = cs_group["volatility_20d"].transform("mean")
    panel["vol_ratio_20d"] = panel["volatility_20d"] / panel["cs_mean_vol20d"].replace(0, np.nan)
    
    # Replace infinite or NaN in new features
    panel["return_20d_rel"] = panel["return_20d_rel"].fillna(0.0)
    panel["vol_ratio_20d"] = panel["vol_ratio_20d"].fillna(1.0).clip(0.1, 10.0)
    
    return panel


REGULARIZED_MARKET_FEATURES = DEFAULT_MARKET_FEATURES + ["return_20d_rel", "vol_ratio_20d"]


class RollingOriginEvaluator:
    """Orchestrates multi-fold expanding-window evaluation on pre-2025 data."""

    def __init__(self, apply_conservative_filing_lag: bool = True):
        self.apply_conservative_lag = apply_conservative_filing_lag
        self.target_builder = TargetBuilder()

    def prepare_dataset(self) -> pd.DataFrame:
        """Loads and constructs full panel with optional conservative filing lag."""
        market_df, fund_df, qual_df = load_raw_data(include_qualitative=True)
        
        # Apply conservative publication lag: if filing occurred on day D,
        # it is only accessible for trading on day D + 1 day
        if self.apply_conservative_lag:
            if fund_df is not None and not fund_df.empty:
                fund_df['filing_date'] = pd.to_datetime(fund_df['filing_date']) + pd.Timedelta(days=1)
            if qual_df is not None and not qual_df.empty:
                qual_df['filing_date'] = pd.to_datetime(qual_df['filing_date']) + pd.Timedelta(days=1)

        # 1. Point in time join
        panel = PointInTimeJoiner.construct_panel(market_df, fund_df, qual_df)
        
        # 2. Build market features (with cross-sectional features)
        panel = build_enhanced_market_features(panel)
        
        # 3. Build fundamental features
        panel = FeatureBuilder.extract_fundamental_features(panel)

        # 4. Build qualitative features
        panel = FeatureBuilder.extract_qualitative_features(panel)
        
        # 5. Forward returns
        panel = self.target_builder.compute_forward_returns(panel)
        
        # Ensure trade_date is datetime
        panel['trade_date_dt'] = pd.to_datetime(panel['trade_date'])
        panel['target_date_dt'] = pd.to_datetime(panel['target_date'])
        
        return panel

    def get_folds(self) -> List[Dict[str, Any]]:
        """
        Defines 4 chronological expanding folds within pre-2025 development data:
        Fold 1: Train 2014-2020, Val 2021
        Fold 2: Train 2014-2021, Val 2022 (Bear market test)
        Fold 3: Train 2014-2022, Val 2023
        Fold 4: Train 2014-2023, Val 2024
        """
        return [
            {
                "fold": 1,
                "train_end": "2020-12-31",
                "val_start": "2021-01-01",
                "val_end": "2021-12-31",
                "name": "Val 2021 (Post-COVID Bull Market)"
            },
            {
                "fold": 2,
                "train_end": "2021-12-31",
                "val_start": "2022-01-01",
                "val_end": "2022-12-31",
                "name": "Val 2022 (Bear Market / Inflation Shock)"
            },
            {
                "fold": 3,
                "train_end": "2022-12-31",
                "val_start": "2023-01-01",
                "val_end": "2023-12-31",
                "name": "Val 2023 (Recovery / Tech Rally)"
            },
            {
                "fold": 4,
                "train_end": "2023-12-31",
                "val_start": "2024-01-01",
                "val_end": "2024-12-31",
                "name": "Val 2024 (Late Development)"
            }
        ]

    def split_fold(
        self,
        panel: pd.DataFrame,
        train_end: str,
        val_start: str,
        val_end: str,
        alpha: float = 0.10
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Splits panel for a fold with temporal purging and train-fitted target labeling."""
        val_start_dt = pd.to_datetime(val_start)
        val_end_dt = pd.to_datetime(val_end)
        
        train = panel[panel['trade_date_dt'] < val_start_dt].copy()
        val = panel[(panel['trade_date_dt'] >= val_start_dt) & (panel['trade_date_dt'] <= val_end_dt)].copy()
        
        # Temporal purging: purge samples whose forward 20-day return window crosses into validation
        train = train[train['target_date_dt'] < val_start_dt].copy()
        
        # Fit threshold on training fold only
        threshold = self.target_builder.fit_threshold(train['forward_return_pct'], alpha)
        
        # Apply labels
        train = self.target_builder.apply_labels(train, threshold)
        val = self.target_builder.apply_labels(val, threshold)
        
        return train, val

    def evaluate_candidates_on_fold(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fits all 5 candidate models exclusively on train_df and evaluates on identical val_df rows.
        """
        train_prev = float(train_df['target_label'].mean())
        val_prev = float(val_df['target_label'].mean())
        y_train = train_df['target_label'].values
        y_val = val_df['target_label'].values

        fold_results = {}

        # -------------------------------------------------------------
        # 1. Existing Market-Only LightGBM (M)
        # -------------------------------------------------------------
        X_tr_m = train_df[DEFAULT_MARKET_FEATURES].copy()
        X_va_m = val_df[DEFAULT_MARKET_FEATURES].copy()
        
        lgb_m = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbosity=-1
        )
        lgb_m.fit(X_tr_m, y_train)
        probs_m = lgb_m.predict_proba(X_va_m)[:, 1]
        fold_results["M"] = self._compute_metrics(y_val, probs_m, train_prev)

        # -------------------------------------------------------------
        # 2. Regularized Logistic Regression Baseline (LogReg)
        # -------------------------------------------------------------
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(imputer.fit_transform(X_tr_m))
        X_va_sc = scaler.transform(imputer.transform(X_va_m))
        
        logreg = LogisticRegression(
            C=0.1,  # Strong L2 regularization
            class_weight="balanced",
            max_iter=500,
            random_state=42
        )
        logreg.fit(X_tr_sc, y_train)
        probs_lr = logreg.predict_proba(X_va_sc)[:, 1]
        fold_results["LogReg"] = self._compute_metrics(y_val, probs_lr, train_prev)

        # -------------------------------------------------------------
        # 3. Carefully Regularized Tree Candidate (M_Regularized)
        # -------------------------------------------------------------
        # Shallow depth (3), high leaf constraint, subsampling, cross-sectional features
        X_tr_reg = train_df[REGULARIZED_MARKET_FEATURES].copy()
        X_va_reg = val_df[REGULARIZED_MARKET_FEATURES].copy()
        
        lgb_reg = lgb.LGBMClassifier(
            n_estimators=80,
            max_depth=3,
            num_leaves=8,
            min_child_samples=80,
            learning_rate=0.03,
            colsample_bytree=0.7,
            subsample=0.8,
            subsample_freq=1,
            reg_alpha=1.0,
            reg_lambda=5.0,
            random_state=42,
            verbosity=-1
        )
        lgb_reg.fit(X_tr_reg, y_train)
        probs_reg = lgb_reg.predict_proba(X_va_reg)[:, 1]
        fold_results["M_Regularized"] = self._compute_metrics(y_val, probs_reg, train_prev)

        # -------------------------------------------------------------
        # 4. Feature Ablation: MF (Market + Fundamentals)
        # -------------------------------------------------------------
        mf_feats = DEFAULT_MARKET_FEATURES + DEFAULT_FUNDAMENTAL_FEATURES
        X_tr_mf = train_df[mf_feats].copy()
        X_va_mf = val_df[mf_feats].copy()
        
        lgb_mf = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbosity=-1
        )
        lgb_mf.fit(X_tr_mf, y_train)
        probs_mf = lgb_mf.predict_proba(X_va_mf)[:, 1]
        fold_results["MF"] = self._compute_metrics(y_val, probs_mf, train_prev)

        # -------------------------------------------------------------
        # 5. Feature Ablation: MFQ (Market + Fundamentals + Qualitative)
        # -------------------------------------------------------------
        mfq_feats = DEFAULT_MARKET_FEATURES + DEFAULT_FUNDAMENTAL_FEATURES + DEFAULT_QUALITATIVE_FEATURES
        X_tr_mfq = train_df[mfq_feats].copy()
        X_va_mfq = val_df[mfq_feats].copy()
        
        lgb_mfq = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbosity=-1
        )
        lgb_mfq.fit(X_tr_mfq, y_train)
        probs_mfq = lgb_mfq.predict_proba(X_va_mfq)[:, 1]
        fold_results["MFQ"] = self._compute_metrics(y_val, probs_mfq, train_prev)

        return fold_results

    def _compute_metrics(self, y_true: np.ndarray, y_prob: np.ndarray, train_prev: float) -> Dict[str, float]:
        roc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.50
        pr = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float(y_true.mean())
        bs = float(brier_score_loss(y_true, y_prob))
        
        # Genuinely deployable Brier skill score: reference is naive constant train prevalence
        bs_ref = float(brier_score_loss(y_true, np.full(len(y_true), train_prev)))
        bss = float(1.0 - (bs / bs_ref)) if bs_ref > 0 else 0.0
        
        ece, mce, _ = compute_calibration_diagnostics(y_true, y_prob, n_bins=10)
        
        return {
            "roc_auc": round(roc, 4),
            "pr_auc": round(pr, 4),
            "brier_score": round(bs, 4),
            "brier_skill": round(bss, 4),
            "ece": round(ece, 4)
        }

    def run_rolling_origin_benchmark(self) -> pd.DataFrame:
        """Executes full rolling-origin evaluation across all folds."""
        logger.info("Starting Rolling-Origin Model Evaluation")
        panel = self.prepare_dataset()
        folds = self.get_folds()
        
        all_fold_rows = []
        
        for f in folds:
            fold_num = f["fold"]
            fold_name = f["name"]
            logger.info("Evaluating fold", fold=fold_num, name=fold_name)
            
            train_df, val_df = self.split_fold(
                panel,
                train_end=f["train_end"],
                val_start=f["val_start"],
                val_end=f["val_end"],
                alpha=0.10
            )
            
            val_prev = float(val_df['target_label'].mean())
            fold_res = self.evaluate_candidates_on_fold(train_df, val_df)
            
            for model_name, metrics in fold_res.items():
                all_fold_rows.append({
                    "fold": fold_num,
                    "val_period": f["val_start"][:4],
                    "val_samples": len(val_df),
                    "val_prevalence": round(val_prev, 4),
                    "model": model_name,
                    **metrics
                })
                
        df_results = pd.DataFrame(all_fold_rows)
        return df_results


if __name__ == "__main__":
    evaluator = RollingOriginEvaluator(apply_conservative_filing_lag=True)
    df_res = evaluator.run_rolling_origin_benchmark()
    
    print("\n=== ROLLING-ORIGIN CROSS-VALIDATION RESULTS (PRE-2025 DEVELOPMENT DATA) ===")
    print(df_res.to_string(index=False))
    
    print("\n=== AGGREGATE MODEL PERFORMANCE ACROSS ALL 4 OUT-OF-TIME FOLDS ===")
    agg_df = df_res.groupby("model").agg(
        mean_roc_auc=("roc_auc", "mean"),
        std_roc_auc=("roc_auc", "std"),
        mean_pr_auc=("pr_auc", "mean"),
        mean_brier=("brier_score", "mean"),
        mean_bss=("brier_skill", "mean"),
        mean_ece=("ece", "mean")
    ).reset_index()
    print(agg_df.sort_values(by="mean_roc_auc", ascending=False).to_string(index=False))
