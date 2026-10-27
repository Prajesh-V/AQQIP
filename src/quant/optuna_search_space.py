"""
AQQIP Optuna Search Space

Defines the hyperparameter search space for LightGBM.
"""
import optuna

def get_lgbm_search_space(trial: optuna.Trial) -> dict:
    """Returns the hyperparameter dictionary for a given trial."""
    return {
        "objective": "binary",
        "metric": "auc", # Use AUC for early stopping
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 300),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "verbose": -1,
        "feature_pre_filter": False,
        "n_jobs": 1  # Required by ADR
    }
