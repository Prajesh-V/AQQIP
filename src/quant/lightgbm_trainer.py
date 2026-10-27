"""
AQQIP LightGBM Trainer

Trains a LightGBM binary classifier with Optuna hyperparameter optimization.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from typing import List, Tuple, Dict, Any

from src.quant.optuna_search_space import get_lgbm_search_space
from src.quant.mlflow_tracking import MLflowTracker
from src.utils.logging import get_logger
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from typing import List, Tuple, Dict, Any, Optional

logger = get_logger("lightgbm_trainer")


class LightGBMTrainer:
    """Trains and evaluates LightGBM models."""

    def __init__(self, features: List[str]):
        self.features = features
        self.tracker = MLflowTracker(experiment_name="AQQIP_Phase3_Quantitative")

    def _evaluate_metrics(self, y_true: pd.Series, y_prob: np.ndarray) -> Dict[str, float]:
        """Calculates classification metrics at standard 0.50 threshold."""
        if len(y_true.unique()) <= 1:
            return {}
            
        y_pred = (y_prob > 0.5).astype(int)
        brier = float(brier_score_loss(y_true, y_prob))
        return {
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
            "pr_auc": float(average_precision_score(y_true, y_prob)),
            "brier_score": brier,
            "brier": brier,
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "positive_rate": float(y_prob.mean())
        }

    @staticmethod
    def select_threshold_on_validation(
        y_val: pd.Series, 
        val_probs: np.ndarray, 
        criterion: str = "max_f1"
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calibrates operational decision threshold strictly on validation partition.
        Criterion: 'max_f1' balances precision and recall for rare drop events.
        """
        thresholds = np.linspace(0.01, 0.90, 90)
        best_thresh = 0.50
        best_score = -1.0
        best_metrics = {}
        
        for t in thresholds:
            preds = (val_probs >= t).astype(int)
            f1 = float(f1_score(y_val, preds, zero_division=0))
            prec = float(precision_score(y_val, preds, zero_division=0))
            rec = float(recall_score(y_val, preds, zero_division=0))
            
            if f1 > best_score:
                best_score = f1
                best_thresh = float(t)
                best_metrics = {
                    "val_threshold": float(t),
                    "val_f1_at_threshold": f1,
                    "val_precision_at_threshold": prec,
                    "val_recall_at_threshold": rec,
                    "val_predicted_positive_rate": float(preds.mean()),
                }
        logger.info(
            "Selected operational probability threshold on validation partition",
            threshold=best_thresh,
            val_f1=best_metrics.get("val_f1_at_threshold"),
            val_precision=best_metrics.get("val_precision_at_threshold"),
            val_recall=best_metrics.get("val_recall_at_threshold")
        )
        return best_thresh, best_metrics

    def train_with_optuna(
        self,
        X_train: pd.DataFrame, y_train: pd.Series,
        X_val: pd.DataFrame, y_val: pd.Series,
        n_trials: int = 20,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        dataset_meta: Optional[Dict[str, Any]] = None
    ) -> Tuple[lgb.Booster, Dict[str, Any]]:
        """Run Optuna study, retrain best model, evaluate holdout and baselines, and log to MLflow."""
        from src.quant.baseline_models import BaselineModels
        
        # Prepare LightGBM datasets
        dtrain = lgb.Dataset(X_train[self.features], label=y_train, params={"feature_pre_filter": False}, free_raw_data=False)
        dval = lgb.Dataset(X_val[self.features], label=y_val, reference=dtrain, params={"feature_pre_filter": False}, free_raw_data=False)

        def objective(trial: optuna.Trial) -> float:
            params = get_lgbm_search_space(trial)
            
            # Using early stopping callback
            callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
            
            model = lgb.train(
                params,
                dtrain,
                valid_sets=[dval],
                num_boost_round=1000,
                callbacks=callbacks
            )
            
            # Predict on validation
            preds = model.predict(X_val[self.features])
            if len(y_val.unique()) > 1:
                return roc_auc_score(y_val, preds)
            return 0.0

        study = optuna.create_study(direction="maximize")
        logger.info(f"Starting Optuna optimization with {n_trials} trials")
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        best_params["objective"] = "binary"
        best_params["verbose"] = -1
        
        logger.info("Optuna finished", best_auc=study.best_value)

        # Retrain on train set with best params
        with self.tracker.start_run(run_name="LightGBM_Purged_Best_Params") as run:
            run_id = run.info.run_id
            
            # Log hyperparameters
            self.tracker.log_params(best_params)
            
            # Log dataset metadata if provided
            if dataset_meta:
                clean_meta = {
                    k: str(v) if not isinstance(v, (int, float, str, bool)) else v
                    for k, v in dataset_meta.items()
                }
                self.tracker.log_params(clean_meta)

            callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
            best_model = lgb.train(
                best_params,
                dtrain,
                valid_sets=[dval],
                num_boost_round=1000,
                callbacks=callbacks
            )
            
            # Validation evaluation
            val_preds = best_model.predict(X_val[self.features])
            val_metrics = self._evaluate_metrics(y_val, val_preds)
            metrics: Dict[str, Any] = {f"val_{k}": v for k, v in val_metrics.items()}
            metrics.update(val_metrics)  # Also keep unprefixed for backward compatibility
            
            # Operational threshold selection on validation partition only
            opt_thresh, opt_val_metrics = self.select_threshold_on_validation(y_val, val_preds, criterion="max_f1")
            metrics.update(opt_val_metrics)
            metrics["operational_threshold_criterion"] = "max_f1_on_validation"
            metrics["operational_threshold"] = opt_thresh
            
            # Test evaluation if provided
            if X_test is not None and y_test is not None and len(X_test) > 0:
                test_preds = best_model.predict(X_test[self.features])
                test_metrics = self._evaluate_metrics(y_test, test_preds)
                for k, v in test_metrics.items():
                    metrics[f"test_{k}"] = v
                
                # Evaluate test set at FROZEN operational threshold calibrated on validation
                test_bin_pred = (test_preds >= opt_thresh).astype(int)
                cm = confusion_matrix(y_test, test_bin_pred)
                tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
                
                metrics["test_operational_threshold"] = opt_thresh
                metrics["test_precision_at_op_threshold"] = float(precision_score(y_test, test_bin_pred, zero_division=0))
                metrics["test_recall_at_op_threshold"] = float(recall_score(y_test, test_bin_pred, zero_division=0))
                metrics["test_f1_at_op_threshold"] = float(f1_score(y_test, test_bin_pred, zero_division=0))
                metrics["test_predicted_positive_rate_at_op_threshold"] = float(test_bin_pred.mean())
                metrics["test_class_prevalence"] = float(y_test.mean())
                metrics["test_confusion_matrix"] = [[tn, fp], [fn, tp]]
                metrics["test_tn"] = float(tn)
                metrics["test_fp"] = float(fp)
                metrics["test_fn"] = float(fn)
                metrics["test_tp"] = float(tp)
                
                # Baseline comparisons on identical test partition
                majority_res = BaselineModels.evaluate_majority_class(y_train, y_test)
                metrics["baseline_majority_roc_auc"] = majority_res.get("roc_auc", 0.5)
                metrics["baseline_majority_pr_auc"] = majority_res.get("pr_auc", float(y_test.mean()))
                metrics["baseline_majority_brier"] = majority_res.get("brier_score", float(y_test.mean()))
                
                lr_res = BaselineModels.evaluate_logistic_regression(
                    X_train, y_train, X_test, y_test, self.features, threshold=opt_thresh
                )
                metrics["baseline_lr_roc_auc"] = lr_res.get("roc_auc", 0.5)
                metrics["baseline_lr_pr_auc"] = lr_res.get("pr_auc", 0.0)
                metrics["baseline_lr_brier"] = lr_res.get("brier_score", 0.0)
                metrics["baseline_lr_f1_at_op_thresh"] = lr_res.get("f1", 0.0)
                metrics["baseline_lr_precision_at_op_thresh"] = lr_res.get("precision", 0.0)
                metrics["baseline_lr_recall_at_op_thresh"] = lr_res.get("recall", 0.0)

            # Log metrics and artifacts
            numeric_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and not np.isnan(v)}
            self.tracker.log_metrics(numeric_metrics)
            self.tracker.log_model(best_model)
            
            metrics["run_id"] = run_id
            return best_model, metrics

