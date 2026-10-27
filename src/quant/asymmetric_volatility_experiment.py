"""
AQQIP Asymmetric Volatility Experiment — Canonical Implementation Module

Implements the preregistered research protocol defined in aqqip_volatility_spec_v4_7.md:
1. Exact candidate feature schemas for Candidates A, B, C, and D.
2. Canonical mathematical formulations for Downside Semivolatility (DSV_20),
   Upside Semivolatility (USV_20), and Downside/Upside Ratio (DAR_20).
3. Chronological rolling-origin partition geometry (Folds 1–4) with strict label-maturity
   purging and independent outer-refit dataset reconstruction.
4. Shared LightGBM training hyperparameters and nested early stopping configuration (no fitting).
5. Deterministic ranking rule ([-predicted_probability, ticker]) and daily alert budget.
6. Paired 20-session within-fold moving-block bootstrap inference engine with exact
   proportion and percentage-point metric equivalences.
7. Decision-matrix evaluation (O-1 through O-9) with strict MECE boundaries.
8. Pre-execution gate validation routines (Gates G1–G7).

This module contains NO LightGBM fitting, refitting, or prediction routines.
"""
from __future__ import annotations

import math
from typing import Dict, Any, List, Tuple, Optional, Sequence
import numpy as np
import pandas as pd

from src.quant.feature_builder import FeatureBuilder
from src.quant.target_builder import TargetBuilder
from src.quant.rolling_origin_evaluator import build_enhanced_market_features

# ==============================================================================
# 1. CANDIDATE FEATURE SCHEMAS (Section 3 of Specification v4.7)
# ==============================================================================

# Candidate A (Baseline, 11 features, preserving audited order)
CANDIDATE_A_FEATURES: Tuple[str, ...] = (
    "return_1d",
    "return_5d",
    "return_20d",
    "return_60d",
    "volatility_20d",
    "daily_range_pct",
    "avg_range_20d",
    "volume_z_score_20d",
    "drawdown_60d",
    "return_20d_rel",
    "vol_ratio_20d",
)

# Candidate B (Additive DSV, 12 features)
CANDIDATE_B_FEATURES: Tuple[str, ...] = CANDIDATE_A_FEATURES + ("DSV_20",)

# Candidate C (Primary substitution: replace volatility_20d with DSV_20, 11 features)
CANDIDATE_C_FEATURES: Tuple[str, ...] = tuple(
    "DSV_20" if feat == "volatility_20d" else feat for feat in CANDIDATE_A_FEATURES
)

# Candidate D (Additive DAR, 12 features)
CANDIDATE_D_FEATURES: Tuple[str, ...] = CANDIDATE_A_FEATURES + ("DAR_20",)

CANDIDATE_SCHEMAS: Dict[str, Tuple[str, ...]] = {
    "A": CANDIDATE_A_FEATURES,
    "B": CANDIDATE_B_FEATURES,
    "C": CANDIDATE_C_FEATURES,
    "D": CANDIDATE_D_FEATURES,
}

DENOMINATOR_FLOOR: float = 1e-4


def get_candidate_features(candidate_name: str) -> List[str]:
    """Returns the ordered list of feature column names for a given candidate."""
    cand = candidate_name.upper().strip()
    if cand not in CANDIDATE_SCHEMAS:
        raise ValueError(
            f"Unknown candidate '{candidate_name}'. Expected one of {list(CANDIDATE_SCHEMAS.keys())}."
        )
    return list(CANDIDATE_SCHEMAS[cand])


# ==============================================================================
# 2. CANONICAL FEATURE FORMULATIONS (Section 3 of Specification v4.7)
# ==============================================================================

def compute_asymmetric_volatility_features(
    df: pd.DataFrame,
    return_col: str = "return_1d",
    price_col: str = "adj_close",
    window: int = 20,
    floor: float = DENOMINATOR_FLOOR,
) -> pd.DataFrame:
    """
    Computes DSV_20, USV_20, and DAR_20 from historical daily returns.

    Formulations:
      DSV_20 = sqrt( (1/20) * sum_{j=0}^{19} min(r_{s,t-j}, 0)^2 )
      USV_20 = sqrt( (1/20) * sum_{j=0}^{19} max(r_{s,t-j}, 0)^2 )
      DAR_20 = DSV_20 / (USV_20 + 1e-4)

    Operates strictly on decimal returns r_{s,t}.
    Strictly backward-looking over [t-19, t]; zero future lookahead.
    """
    out = df.copy()

    # Ensure sorted by ticker and trade_date for correct rolling operations
    if "ticker" in out.columns and "trade_date" in out.columns:
        out = out.sort_values(["ticker", "trade_date"]).reset_index(drop=True)

    # Compute daily return if not present
    if return_col not in out.columns:
        if price_col not in out.columns:
            raise KeyError(f"Neither '{return_col}' nor '{price_col}' found in DataFrame.")
        if "ticker" in out.columns:
            out[return_col] = out.groupby("ticker")[price_col].pct_change(periods=1)
        else:
            out[return_col] = out[price_col].pct_change(periods=1)

    # Squared negative returns and positive returns
    neg_sq = np.minimum(out[return_col], 0.0) ** 2
    pos_sq = np.maximum(out[return_col], 0.0) ** 2

    # 20-session rolling mean of squared semireturns
    if "ticker" in out.columns:
        dsv_sq_mean = neg_sq.groupby(out["ticker"]).transform(
            lambda s: s.rolling(window, min_periods=window).mean()
        )
        usv_sq_mean = pos_sq.groupby(out["ticker"]).transform(
            lambda s: s.rolling(window, min_periods=window).mean()
        )
    else:
        dsv_sq_mean = neg_sq.rolling(window, min_periods=window).mean()
        usv_sq_mean = pos_sq.rolling(window, min_periods=window).mean()

    out["DSV_20"] = np.sqrt(dsv_sq_mean)
    out["USV_20"] = np.sqrt(usv_sq_mean)
    out["DAR_20"] = out["DSV_20"] / (out["USV_20"] + floor)

    return out


def build_candidate_panel(raw_ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the complete multi-candidate panel from raw OHLCV prices:
    1. Baseline market features + cross-sectional features (return_20d_rel, vol_ratio_20d).
    2. Asymmetric volatility features (DSV_20, USV_20, DAR_20).
    3. Forward returns (forward_return_pct) via TargetBuilder (H=20).
    4. Observed global trading-session index s in [1, 2768].
    """
    # 1. Enhanced baseline market features (11 baseline features)
    panel = build_enhanced_market_features(raw_ohlcv_df)

    # 2. Add canonical asymmetric volatility features
    panel = compute_asymmetric_volatility_features(panel, return_col="return_1d", price_col="adj_close")

    # 3. Compute forward returns via existing TargetBuilder
    tb = TargetBuilder()
    panel = tb.compute_forward_returns(panel)

    # 4. Global session index
    all_dates = sorted(panel["trade_date"].unique())
    session_map = {d: idx + 1 for idx, d in enumerate(all_dates)}
    panel["session"] = panel["trade_date"].map(session_map)

    return panel


# ==============================================================================
# 3. PARTITION GEOMETRY & PURGING (Section 6 of Specification v4.7)
# ==============================================================================

FOLD_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "fold": 1,
        "eval_year": 2021,
        "inner_train": (1, 1490),
        "inner_val": (1511, 1743),
        "outer_refit": (1, 1743),
        "eval_retained": (1764, 1995),
        "eval_trimmed": (1996, 2015),
        "expected_counts": {
            "inner_train": 47680,
            "inner_val": 7456,
            "outer_refit": 55776,
            "eval_retained": 7424,
            "eval_trimmed": 640,
        },
    },
    {
        "fold": 2,
        "eval_year": 2022,
        "inner_train": (1, 1743),
        "inner_val": (1764, 1995),
        "outer_refit": (1, 1995),
        "eval_retained": (2016, 2246),
        "eval_trimmed": (2247, 2266),
        "expected_counts": {
            "inner_train": 55776,
            "inner_val": 7424,
            "outer_refit": 63840,
            "eval_retained": 7392,
            "eval_trimmed": 640,
        },
    },
    {
        "fold": 3,
        "eval_year": 2023,
        "inner_train": (1, 1995),
        "inner_val": (2016, 2246),
        "outer_refit": (1, 2246),
        "eval_retained": (2267, 2496),
        "eval_trimmed": (2497, 2516),
        "expected_counts": {
            "inner_train": 63840,
            "inner_val": 7392,
            "outer_refit": 71872,
            "eval_retained": 7360,
            "eval_trimmed": 640,
        },
    },
    {
        "fold": 4,
        "eval_year": 2024,
        "inner_train": (1, 2246),
        "inner_val": (2267, 2496),
        "outer_refit": (1, 2496),
        "eval_retained": (2517, 2748),
        "eval_trimmed": (2749, 2768),
        "expected_counts": {
            "inner_train": 71872,
            "inner_val": 7360,
            "outer_refit": 79872,
            "eval_retained": 7424,
            "eval_trimmed": 640,
        },
    },
]


def split_experiment_fold(
    panel: pd.DataFrame,
    fold_num: int,
    alpha: float = 0.10,
) -> Dict[str, Any]:
    """
    Constructs the exact partitions for a fold with independent threshold calibration
    and strict label-maturity purging.

    Key Invariants:
    1. tau_inner is calibrated strictly on inner_train.
    2. tau_inner assigns binary target_label to both inner_train and inner_val.
    3. outer_refit is reconstructed independently from the historical panel (NOT by
       concatenating pre-trimmed inner splits).
    4. tau_outer is calibrated strictly on outer_refit.
    5. tau_outer assigns binary target_label to outer_refit and outer_eval.
    6. outer_eval is strictly retained evaluation observations (terminal trim excluded).
    """
    if fold_num not in {1, 2, 3, 4}:
        raise ValueError(f"Invalid fold_num {fold_num}. Must be in {{1, 2, 3, 4}}.")

    if "session" not in panel.columns:
        raise KeyError("Panel DataFrame must contain a 'session' column.")

    fold_def = FOLD_DEFINITIONS[fold_num - 1]
    tb = TargetBuilder()

    # 1. Slice partitions by session index
    s_in_tr_start, s_in_tr_end = fold_def["inner_train"]
    s_in_val_start, s_in_val_end = fold_def["inner_val"]
    s_refit_start, s_refit_end = fold_def["outer_refit"]
    s_eval_start, s_eval_end = fold_def["eval_retained"]

    inner_train_raw = panel[
        (panel["session"] >= s_in_tr_start) & (panel["session"] <= s_in_tr_end)
    ].copy()
    inner_val_raw = panel[
        (panel["session"] >= s_in_val_start) & (panel["session"] <= s_in_val_end)
    ].copy()
    outer_refit_raw = panel[
        (panel["session"] >= s_refit_start) & (panel["session"] <= s_refit_end)
    ].copy()
    outer_eval_raw = panel[
        (panel["session"] >= s_eval_start) & (panel["session"] <= s_eval_end)
    ].copy()

    # 2. Inner-threshold calibration (tau_inner) on inner_train
    tau_inner = tb.fit_threshold(inner_train_raw["forward_return_pct"], alpha)
    inner_train = tb.apply_labels(inner_train_raw, tau_inner)
    inner_val = tb.apply_labels(inner_val_raw, tau_inner)

    # 3. Outer-threshold calibration (tau_outer) on outer_refit
    tau_outer = tb.fit_threshold(outer_refit_raw["forward_return_pct"], alpha)
    outer_refit = tb.apply_labels(outer_refit_raw, tau_outer)
    outer_eval = tb.apply_labels(outer_eval_raw, tau_outer)

    return {
        "fold": fold_num,
        "eval_year": fold_def["eval_year"],
        "tau_inner": tau_inner,
        "tau_outer": tau_outer,
        "inner_train": inner_train,
        "inner_val": inner_val,
        "outer_refit": outer_refit,
        "outer_eval": outer_eval,
    }


# ==============================================================================
# 4. PREREGISTERED MODEL CONFIGURATION (Section 4 of Specification v4.7)
# ==============================================================================

BASE_LGBM_HYPERPARAMS: Dict[str, Any] = {
    "max_depth": 3,
    "num_leaves": 8,
    "min_child_samples": 80,
    "learning_rate": 0.03,
    "colsample_bytree": 0.70,
    "subsample": 0.80,
    "subsample_freq": 1,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
    "random_state": 42,
    "verbosity": -1,
    "objective": "binary",
    "metric": "binary_logloss",
}

TRAINING_POLICY: Dict[str, Any] = {
    "max_boosting_rounds": 300,
    "early_stopping_patience": 25,
    "eval_metric": "binary_logloss",
}


# ==============================================================================
# 5. DETERMINISTIC RANKING & ALERT BUDGET (Section 7 of Specification v4.7)
# ==============================================================================

def compute_daily_alert_budget(n_securities: int, fraction: float = 0.05) -> int:
    """
    Computes daily alert budget:
      k_t = max(1, round(N_t * 0.05))
    For N_t = 32: max(1, round(1.6)) = 2.
    """
    return max(1, int(round(n_securities * fraction)))


def deterministic_rank(
    df: pd.DataFrame,
    prob_col: str = "predicted_probability",
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    """
    Deterministic ranking rule:
      Sort by (-predicted_probability, ticker ascending).
    No return-based key permitted.
    """
    return df.sort_values(
        by=[prob_col, ticker_col],
        ascending=[False, True],
    ).reset_index(drop=True)


def assign_daily_alerts(
    df: pd.DataFrame,
    prob_col: str = "predicted_probability",
    ticker_col: str = "ticker",
    date_col: str = "trade_date",
    alert_col: str = "alert",
    fraction: float = 0.05,
) -> pd.DataFrame:
    """
    Applies the deterministic ranking rule and top-5% alert budget per trade date.
    Returns a DataFrame with a binary alert column (1 for top k_t, 0 otherwise).
    """
    out_list = []
    for dt, group in df.groupby(date_col):
        ranked = deterministic_rank(group, prob_col=prob_col, ticker_col=ticker_col)
        k_t = compute_daily_alert_budget(len(ranked), fraction=fraction)
        ranked[alert_col] = 0
        ranked.iloc[:k_t, ranked.columns.get_loc(alert_col)] = 1
        out_list.append(ranked)

    return pd.concat(out_list, ignore_index=True)


# ==============================================================================
# 6. BOOTSTRAP INFERENCE & METRIC EQUIVALENCES (Section 8 of Specification v4.7)
# ==============================================================================

def proportion_to_pp(val: float) -> float:
    """Converts a decimal proportion difference to percentage points (x 100.0)."""
    return val * 100.0


def pp_to_proportion(val: float) -> float:
    """Converts percentage points to a decimal proportion (/ 100.0)."""
    return val / 100.0


def run_moving_block_bootstrap(
    fold_eval_dfs: List[pd.DataFrame],
    block_length: int = 20,
    n_replicates: int = 1000,
    seed: int = 20240101,
    max_retries: int = 10000,
    max_attempts: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Executes the paired 20-session within-fold moving-block bootstrap inference
    evaluating Candidate C versus Baseline A under strict rejection resampling.

    Guarantees:
    - Blocks never cross fold boundaries.
    - Both candidates evaluated on identical sampled date-security pairs.
    - Computes both raw proportion differences and converted percentage-point differences.
    - Undefined metric handling (Rejection Resampling):
      * If a sampled, fold-aggregated bootstrap replicate contains zero actual positive
        outcomes (tp + fn == 0), recall is undefined.
      * If either candidate has zero alerts (tp + fp == 0), false-positive share is undefined.
      * Replicates with undefined statistics are rejected and redrawn deterministically
        from the RNG stream until n_replicates valid replicates are collected.
      * If the retry limit (max_retries / max_attempts) is exceeded, raises RuntimeError.
      * Never silently substitutes artificial zero differences or NaN/inf values.
    """
    if n_replicates <= 0:
        raise ValueError("n_replicates must be a positive integer.")
    if block_length <= 0:
        raise ValueError("block_length must be a positive integer.")
    if not fold_eval_dfs:
        raise ValueError("fold_eval_dfs cannot be empty.")

    rng = np.random.default_rng(seed)

    # Pre-extract dates and grouped data per fold
    fold_data = []
    for df in fold_eval_dfs:
        dates = sorted(df["trade_date"].unique())
        # Map date -> subset of observations
        date_groups = {d: group for d, group in df.groupby("trade_date")}
        fold_data.append((dates, len(dates), date_groups))

    diff_fp_share_prop = np.empty(n_replicates, dtype=np.float64)
    diff_recall_prop = np.empty(n_replicates, dtype=np.float64)

    accepted = 0
    attempt = 0
    n_rejected = 0
    rejection_breakdown = {"zero_positives": 0, "zero_alerts": 0}

    effective_max_attempts = (
        max_attempts if max_attempts is not None else (n_replicates + max_retries)
    )

    while accepted < n_replicates:
        if attempt >= effective_max_attempts:
            raise RuntimeError(
                f"Bootstrap retry limit exceeded: attempted {attempt} draws without collecting "
                f"{n_replicates} valid replicates (accepted: {accepted}, rejected: {n_rejected}). "
                f"Rejection breakdown: {rejection_breakdown}."
            )

        attempt += 1

        tp_A = fp_A = tn_A = fn_A = 0
        tp_C = fp_C = tn_C = fn_C = 0

        for dates, T_k, date_groups in fold_data:
            n_blocks = math.ceil(T_k / block_length)
            max_start = max(0, T_k - block_length)
            starts = rng.integers(0, max_start + 1, size=n_blocks)

            sampled_dates = []
            for start in starts:
                sampled_dates.extend(dates[start : start + block_length])
            sampled_dates = sampled_dates[:T_k]

            for d in sampled_dates:
                grp = date_groups[d]
                y = grp["target_label"].values
                a_A = grp["alert_A"].values
                a_C = grp["alert_C"].values

                # Vectorized confusion-matrix accumulation
                tp_A += int(np.sum((a_A == 1) & (y == 1)))
                fp_A += int(np.sum((a_A == 1) & (y == 0)))
                tn_A += int(np.sum((a_A == 0) & (y == 0)))
                fn_A += int(np.sum((a_A == 0) & (y == 1)))

                tp_C += int(np.sum((a_C == 1) & (y == 1)))
                fp_C += int(np.sum((a_C == 1) & (y == 0)))
                tn_C += int(np.sum((a_C == 0) & (y == 0)))
                fn_C += int(np.sum((a_C == 0) & (y == 1)))

        denom_rec_A = tp_A + fn_A
        denom_rec_C = tp_C + fn_C
        denom_fp_A = fp_A + tp_A
        denom_fp_C = fp_C + tp_C

        # Validity check 1: Recall denominator must be strictly positive
        if denom_rec_A <= 0 or denom_rec_C <= 0:
            n_rejected += 1
            rejection_breakdown["zero_positives"] += 1
            continue

        # Validity check 2: Alert denominators must be strictly positive for both candidates
        if denom_fp_A <= 0 or denom_fp_C <= 0:
            n_rejected += 1
            rejection_breakdown["zero_alerts"] += 1
            continue

        # Both metrics strictly defined — compute proportions without zero substitutions
        fp_share_A = fp_A / denom_fp_A
        fp_share_C = fp_C / denom_fp_C
        rec_A = tp_A / denom_rec_A
        rec_C = tp_C / denom_rec_C

        diff_fp_share_prop[accepted] = fp_share_C - fp_share_A
        diff_recall_prop[accepted] = rec_C - rec_A
        accepted += 1

    # Convert replicate arrays to percentage points (difference in percentage points)
    diff_fp_share_pp = diff_fp_share_prop * 100.0
    diff_recall_pp = diff_recall_prop * 100.0

    # Quantiles using standard linear interpolation
    ci_fp_prop = [
        float(np.quantile(diff_fp_share_prop, 0.025, method="linear")),
        float(np.quantile(diff_fp_share_prop, 0.975, method="linear")),
    ]
    ci_fp_pp = [
        float(np.quantile(diff_fp_share_pp, 0.025, method="linear")),
        float(np.quantile(diff_fp_share_pp, 0.975, method="linear")),
    ]

    lb_recall_prop = float(np.quantile(diff_recall_prop, 0.05, method="linear"))
    lb_recall_pp = float(np.quantile(diff_recall_pp, 0.05, method="linear"))

    return {
        "diff_fp_share_prop": diff_fp_share_prop,
        "diff_recall_prop": diff_recall_prop,
        "diff_fp_share_pp": diff_fp_share_pp,
        "diff_recall_pp": diff_recall_pp,
        "ci_fp_share_prop": ci_fp_prop,
        "ci_fp_share_pp": ci_fp_pp,
        "lb_recall_prop": lb_recall_prop,
        "lb_recall_pp": lb_recall_pp,
        "n_accepted": accepted,
        "n_rejected": n_rejected,
        "n_attempted": attempt,
        "n_replicates": n_replicates,
        "rejection_breakdown": rejection_breakdown,
    }


# ==============================================================================
# 7. DECISION MATRIX EVALUATION (Section 10 of Specification v4.7)
# ==============================================================================

def evaluate_decision_matrix(
    delta_fp_share_pp: float,
    ci_fp_upper_pp: float,
    lb_recall_pp: float,
    rho_upside: float,
    d_fp_count: int,
) -> Dict[str, Any]:
    """
    Evaluates Candidate C vs. Baseline A empirical results against the
    mutually exclusive, collectively exhaustive (MECE) Decision Matrix.

    Inputs (in percentage points / percentages):
      delta_fp_share_pp: Point estimate of FP share difference (pp).
      ci_fp_upper_pp: 97.5th percentile upper bound of FP share difference (pp).
      lb_recall_pp: 5th percentile lower bound of recall difference (pp).
      rho_upside: Percentage of displaced false positives with positive returns (%).
      d_fp_count: Count of displaced false positives |D_FP|.

    Governing Invariant:
      Operational status under ALL outcomes leaves Baseline A unchanged.
    """
    # Criteria evaluations
    recall_ni_pass = lb_recall_pp >= -1.0  # exact -1.0 pp is PASS
    ci_pass = ci_fp_upper_pp < 0.0  # strictly < 0.0 pp
    mech_pass = (rho_upside > 60.0) and (d_fp_count > 0)  # strictly > 60.0%

    # Tier 1: delta_fp_share_pp <= -4.0 pp (exact -4.0 pp is Tier 1)
    if delta_fp_share_pp <= -4.0:
        if not recall_ni_pass:
            return {
                "outcome_id": "O-4",
                "tier": 1,
                "recall_ni": "FAIL",
                "ci_criterion": "PASS" if ci_pass else "FAIL",
                "mechanism": "PASS" if mech_pass else "FAIL",
                "status": "Reject Candidate C. (Baseline A unchanged)",
                "interpretation": "False-positive reduction achieved via excessive recall degradation.",
            }
        if not ci_pass:
            return {
                "outcome_id": "O-3",
                "tier": 1,
                "recall_ni": "PASS",
                "ci_criterion": "FAIL",
                "mechanism": "PASS" if mech_pass else "FAIL",
                "status": "Retain as high-priority research. (Baseline A unchanged)",
                "interpretation": "Point target met; confidence interval fails to exclude zero at 95% confidence.",
            }
        if mech_pass:
            return {
                "outcome_id": "O-1",
                "tier": 1,
                "recall_ni": "PASS",
                "ci_criterion": "PASS",
                "mechanism": "PASS",
                "status": "Preregistered criteria met; eligible for separate research review. (Baseline A unchanged)",
                "interpretation": "Target effect, statistical confidence, safety, and mechanism all supported.",
            }
        else:
            return {
                "outcome_id": "O-2",
                "tier": 1,
                "recall_ni": "PASS",
                "ci_criterion": "PASS",
                "mechanism": "FAIL",
                "status": "Provisional hold / mechanism audit. (Baseline A unchanged)",
                "interpretation": "Statistical criteria met; pruning mechanism deviates from downside-volatility hypothesis.",
            }

    # Tier 2: -4.0 pp < delta_fp_share_pp < -1.0 pp
    elif delta_fp_share_pp < -1.0:
        if not recall_ni_pass:
            return {
                "outcome_id": "O-7",
                "tier": 2,
                "recall_ni": "FAIL",
                "ci_criterion": "PASS" if ci_pass else "FAIL",
                "mechanism": "PASS" if mech_pass else "FAIL",
                "status": "Reject Candidate C. (Baseline A unchanged)",
                "interpretation": "Sub-target effect accompanied by unacceptable recall degradation.",
            }
        if ci_pass:
            return {
                "outcome_id": "O-5",
                "tier": 2,
                "recall_ni": "PASS",
                "ci_criterion": "PASS",
                "mechanism": "PASS" if mech_pass else "FAIL",
                "status": "Sub-target significant signal; archive for future research review. (Baseline A unchanged)",
                "interpretation": "Statistically detectable improvement below the 4.0 pp practical hurdle.",
            }
        else:
            return {
                "outcome_id": "O-6",
                "tier": 2,
                "recall_ni": "PASS",
                "ci_criterion": "FAIL",
                "mechanism": "PASS" if mech_pass else "FAIL",
                "status": "Inconclusive / insufficient evidence. (Baseline A unchanged)",
                "interpretation": "Sub-target point reduction not statistically confirmed.",
            }

    # Tier 3A & 3B: delta_fp_share_pp >= -1.0 pp
    else:
        if not recall_ni_pass:
            return {
                "outcome_id": "O-9",
                "tier": 3,
                "recall_ni": "FAIL",
                "ci_criterion": "PASS" if ci_pass else "FAIL",
                "mechanism": "PASS" if mech_pass else "FAIL",
                "status": "Evidence against Candidate C. (Baseline A unchanged)",
                "interpretation": "Recall degradation without meaningful false-positive reduction.",
            }

        # delta_fp_share_pp in [-1.0, 0.0)
        if delta_fp_share_pp < 0.0:
            if ci_pass:
                return {
                    "outcome_id": "O-8A1",
                    "tier": 3,
                    "recall_ni": "PASS",
                    "ci_criterion": "PASS",
                    "mechanism": "PASS" if mech_pass else "FAIL",
                    "status": "Small, below-target improvement (statistically detectable). (Baseline A unchanged)",
                    "interpretation": "Statistically detectable false-positive reduction, but below 4.0 pp hurdle.",
                }
            else:
                return {
                    "outcome_id": "O-8A2",
                    "tier": 3,
                    "recall_ni": "PASS",
                    "ci_criterion": "FAIL",
                    "mechanism": "PASS" if mech_pass else "FAIL",
                    "status": "Small, below-target improvement (inconclusive). (Baseline A unchanged)",
                    "interpretation": "Small point reduction not statistically distinguishable from zero at 95% confidence.",
                }
        # delta_fp_share_pp >= 0.0 pp
        else:
            return {
                "outcome_id": "O-8B",
                "tier": 3,
                "recall_ni": "PASS",
                "ci_criterion": "FAIL",
                "mechanism": "PASS" if mech_pass else "FAIL",
                "status": "Zero improvement or deterioration. (Baseline A unchanged)",
                "interpretation": "Candidate C produces equal or worse false-positive share.",
            }


# ==============================================================================
# 8. HARD PRE-EXECUTION GATE VALIDATION (Section 5.3 of Specification v4.7)
# ==============================================================================

def validate_gate_g1(eval_panel_df: pd.DataFrame) -> Tuple[bool, str]:
    """G1: All 32 securities present on each of the 925 retained evaluation dates."""
    dates = eval_panel_df["trade_date"].unique()
    if len(dates) != 925:
        return False, f"Expected 925 evaluation dates, found {len(dates)}."
    counts = eval_panel_df.groupby("trade_date")["ticker"].count()
    if not (counts == 32).all():
        return False, f"Date ticker counts deviate from 32 (range: {counts.min()} to {counts.max()})."
    return True, f"All 925 evaluation dates have exactly 32 tickers (29,600 rows)."


def validate_gate_g2(eval_panel_df: pd.DataFrame) -> Tuple[bool, str]:
    """G2: All features for A, B, C, D complete, finite (no NaN, no inf) across eval panel."""
    all_features = sorted(
        list(
            set(
                CANDIDATE_A_FEATURES
                + CANDIDATE_B_FEATURES
                + CANDIDATE_C_FEATURES
                + CANDIDATE_D_FEATURES
            )
        )
    )
    for feat in all_features:
        if feat not in eval_panel_df.columns:
            return False, f"Feature '{feat}' missing from evaluation panel."

    sub = eval_panel_df[all_features]
    nan_count = int(sub.isna().sum().sum())
    inf_count = int(np.isinf(sub.select_dtypes(include=np.number)).sum().sum())

    if nan_count > 0 or inf_count > 0:
        return False, f"Evaluation features contain {nan_count} NaNs and {inf_count} infs."
    return True, f"All {len(all_features)} candidate features are complete and finite across 29,600 rows."


def validate_gate_g3(eval_panel_df: pd.DataFrame) -> Tuple[bool, str]:
    """G3: forward_return_pct non-NaN and finite for all 29,600 retained evaluation rows."""
    if "forward_return_pct" not in eval_panel_df.columns:
        return False, "Target field 'forward_return_pct' missing from evaluation panel."

    returns = eval_panel_df["forward_return_pct"]
    nan_count = int(returns.isna().sum())
    inf_count = int(np.isinf(returns).sum())

    if nan_count > 0 or inf_count > 0:
        return False, f"forward_return_pct contains {nan_count} NaNs and {inf_count} infs."
    return True, f"forward_return_pct is valid and finite for all {len(eval_panel_df)} evaluation rows."


def validate_gate_g4(eval_panel_df: pd.DataFrame) -> Tuple[bool, str]:
    """G4: Strict shared universe across Candidates A, B, C, D."""
    for cand_name, feats in CANDIDATE_SCHEMAS.items():
        sub = eval_panel_df.dropna(subset=list(feats))
        if len(sub) != len(eval_panel_df):
            return False, f"Candidate {cand_name} drops {len(eval_panel_df) - len(sub)} rows due to NaNs."
    return True, "All four candidates evaluate the identical 29,600 date-security observation universe."


def validate_gate_g5(full_panel_df: pd.DataFrame) -> Tuple[bool, str]:
    """G5: Row-count and partition reconciliation across Folds 1–4."""
    for fold_def in FOLD_DEFINITIONS:
        k = fold_def["fold"]
        exp = fold_def["expected_counts"]

        in_tr = full_panel_df[
            (full_panel_df["session"] >= fold_def["inner_train"][0])
            & (full_panel_df["session"] <= fold_def["inner_train"][1])
        ]
        in_val = full_panel_df[
            (full_panel_df["session"] >= fold_def["inner_val"][0])
            & (full_panel_df["session"] <= fold_def["inner_val"][1])
        ]
        refit = full_panel_df[
            (full_panel_df["session"] >= fold_def["outer_refit"][0])
            & (full_panel_df["session"] <= fold_def["outer_refit"][1])
        ]
        ev_ret = full_panel_df[
            (full_panel_df["session"] >= fold_def["eval_retained"][0])
            & (full_panel_df["session"] <= fold_def["eval_retained"][1])
        ]

        if len(in_tr) != exp["inner_train"]:
            return False, f"Fold {k} inner_train count mismatch: {len(in_tr)} vs {exp['inner_train']}."
        if len(in_val) != exp["inner_val"]:
            return False, f"Fold {k} inner_val count mismatch: {len(in_val)} vs {exp['inner_val']}."
        if len(refit) != exp["outer_refit"]:
            return False, f"Fold {k} outer_refit count mismatch: {len(refit)} vs {exp['outer_refit']}."
        if len(ev_ret) != exp["eval_retained"]:
            return False, f"Fold {k} eval_retained count mismatch: {len(ev_ret)} vs {exp['eval_retained']}."

    return True, "All inner-training, inner-validation, outer-refit, and eval row counts match v4.7 exactly."


def validate_gate_g6(full_panel_df: pd.DataFrame) -> Tuple[bool, str]:
    """
    G6: Expected vs. unexpected missingness.
    - Zero NaNs for s >= 61 across candidate features.
    - Zero non-finite values (inf / -inf) across entire panel.
    - Exactly 1,920 baseline rows with expected warm-up NaNs in s in [1, 60].
    - Zero NaNs for DSV_20 and DAR_20 for s >= 21.
    """
    all_features = sorted(
        list(
            set(
                CANDIDATE_A_FEATURES
                + CANDIDATE_B_FEATURES
                + CANDIDATE_C_FEATURES
                + CANDIDATE_D_FEATURES
            )
        )
    )
    # Check for infinite values across all sessions
    inf_count = int(np.isinf(full_panel_df[all_features].select_dtypes(include=np.number)).sum().sum())
    if inf_count > 0:
        return False, f"Panel contains {inf_count} non-finite values (inf/-inf)."

    # Post-warmup check (s >= 61)
    post_warmup = full_panel_df[full_panel_df["session"] >= 61]
    post_nans = int(post_warmup[all_features].isna().sum().sum())
    if post_nans > 0:
        return False, f"Unexpected missingness post-session 60: {post_nans} NaNs."

    # Warmup check for baseline (s in [1, 60])
    warmup = full_panel_df[full_panel_df["session"] <= 60]
    warmup_nans = int(warmup[list(CANDIDATE_A_FEATURES)].isna().any(axis=1).sum())
    if warmup_nans != 1920:
        return False, f"Expected 1,920 baseline warm-up rows with NaN, found {warmup_nans}."

    # DSV / DAR post-session 20 check
    post_20 = full_panel_df[full_panel_df["session"] >= 21]
    dsv_nans = int(post_20["DSV_20"].isna().sum())
    dar_nans = int(post_20["DAR_20"].isna().sum())
    if dsv_nans > 0 or dar_nans > 0:
        return False, f"DSV/DAR contain NaNs post-session 20: {dsv_nans}/{dar_nans}."

    return True, "Expected training warm-up missingness confirmed; zero unexpected missingness or infs."


def validate_gate_g7(full_panel_df: pd.DataFrame) -> Tuple[bool, str]:
    """G7: Warm-up containment invariant: feature-incomplete rows strictly in training."""
    min_val_session = min(
        [f["inner_val"][0] for f in FOLD_DEFINITIONS]
        + [f["eval_retained"][0] for f in FOLD_DEFINITIONS]
    )
    if min_val_session <= 60:
        return False, f"Validation session s={min_val_session} overlaps with warm-up window s in [1, 60]."
    return True, f"Earliest validation/evaluation session is s={min_val_session} (well beyond s=60)."
