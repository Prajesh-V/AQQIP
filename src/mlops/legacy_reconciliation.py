"""
AQQIP Legacy Prediction Target Date Reconciliation Module

Audits and reconciles historical prediction records against the frozen target contract:
The twentieth subsequent observed trading session in the historical OHLCV record.

Flags nonconforming legacy records with status 'invalid_target', preserving the original
prediction values, timestamps, and model run IDs for auditability.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from typing import Dict, List, Any, Optional
import mysql.connector
import pandas as pd

from config.settings import get_settings
from src.database.mysql_client import get_mysql_connection
from src.utils.logging import get_logger

logger = get_logger("legacy_reconciliation")


class LegacyReconciliationAuditor:
    """Audits and reconciles prediction records against the 20-trading-day contract."""

    def __init__(self):
        self.settings = get_settings().mysql

    def fetch_trading_calendars(self) -> Dict[str, List[date]]:
        """Loads sorted list of trading dates per ticker from MySQL."""
        with get_mysql_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT ticker, trade_date FROM ohlcv_prices ORDER BY ticker, trade_date ASC"
            )
            calendars: Dict[str, List[date]] = {}
            for t, d in cursor.fetchall():
                calendars.setdefault(t, []).append(d)
            return calendars

    def fetch_all_predictions(self) -> List[Dict[str, Any]]:
        """Fetches all rows from predictions table."""
        with get_mysql_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM predictions ORDER BY id ASC")
            return cursor.fetchall()

    def audit_records(self) -> List[Dict[str, Any]]:
        """Audits every prediction against the frozen target contract."""
        calendars = self.fetch_trading_calendars()
        predictions = self.fetch_all_predictions()
        audit_results: List[Dict[str, Any]] = []

        for p in predictions:
            pred_id = p["id"]
            ticker = p["ticker"]
            p_dt = p["predicted_at"]
            pred_date = p_dt.date() if isinstance(p_dt, datetime) else p_dt
            recorded_target = p["target_date"]
            status = p["status"]

            t_dates = calendars.get(ticker, [])

            if not t_dates:
                # Synthetic or unmonitored ticker (e.g. TESTML)
                audit_results.append({
                    "id": pred_id,
                    "ticker": ticker,
                    "model_run_id": p["model_run_id"],
                    "predicted_at": str(pred_date),
                    "recorded_target": str(recorded_target),
                    "contract_target": None,
                    "discrepancy_days": None,
                    "is_conforming": False,
                    "reason": "Ticker not present in OHLCV market dataset (synthetic/test record)",
                    "prior_status": status,
                    "recommended_status": "invalid_target" if status != "invalid_target" else status,
                    "actual_price_change_pct": float(p["actual_price_change_pct"]) if p.get("actual_price_change_pct") is not None else None,
                })
                continue

            if pred_date in t_dates:
                pos = t_dates.index(pred_date)
                if pos + 20 < len(t_dates):
                    contract_target = t_dates[pos + 20]
                    is_conforming = (recorded_target == contract_target)
                    delta_days = (recorded_target - contract_target).days
                    reason = (
                        "Conforms exactly to 20th subsequent trading session contract"
                        if is_conforming
                        else f"Target date off by {delta_days} calendar days from 20th trading session ({contract_target})"
                    )
                    recommended_status = status if is_conforming else "invalid_target"
                else:
                    # Beyond available OHLCV (genuinely prospective)
                    contract_target = None
                    is_conforming = (status == "open")
                    delta_days = None
                    reason = "Prospective prediction (20th trading session is in future beyond OHLCV edge)"
                    recommended_status = "open"
            else:
                contract_target = None
                is_conforming = False
                delta_days = None
                reason = f"Predicted date {pred_date} not an observed trading session in OHLCV"
                recommended_status = "invalid_target"

            audit_results.append({
                "id": pred_id,
                "ticker": ticker,
                "model_run_id": p["model_run_id"],
                "predicted_at": str(pred_date),
                "recorded_target": str(recorded_target),
                "contract_target": str(contract_target) if contract_target else None,
                "discrepancy_days": delta_days,
                "is_conforming": is_conforming,
                "reason": reason,
                "prior_status": status,
                "recommended_status": recommended_status,
                "actual_price_change_pct": float(p["actual_price_change_pct"]) if p.get("actual_price_change_pct") is not None else None,
            })

        return audit_results

    def apply_reconciliation(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Flags nonconforming records as 'invalid_target' in MySQL without deleting rows.
        Leaves valid prospective and retrospective records intact.
        """
        audit_results = self.audit_records()
        flagged_ids = []

        for item in audit_results:
            if not item["is_conforming"] and item["prior_status"] != "open" and item["prior_status"] != "invalid_target":
                flagged_ids.append(item["id"])

        if not dry_run and flagged_ids:
            with get_mysql_connection() as conn:
                cursor = conn.cursor()
                format_strings = ','.join(['%s'] * len(flagged_ids))
                cursor.execute(
                    f"UPDATE predictions SET status = 'invalid_target' WHERE id IN ({format_strings})",
                    tuple(flagged_ids)
                )
                conn.commit()
                logger.info(
                    "Updated nonconforming predictions to invalid_target",
                    count=len(flagged_ids),
                    ids=flagged_ids
                )

        return {
            "total_audited": len(audit_results),
            "conforming_count": sum(1 for x in audit_results if x["is_conforming"]),
            "nonconforming_count": sum(1 for x in audit_results if not x["is_conforming"]),
            "flagged_ids": flagged_ids,
            "dry_run": dry_run,
            "audit_results": audit_results
        }


if __name__ == "__main__":
    auditor = LegacyReconciliationAuditor()
    report = auditor.apply_reconciliation(dry_run=False)
    print(f"Total audited: {report['total_audited']}")
    print(f"Conforming count: {report['conforming_count']}")
    print(f"Non-conforming count: {report['nonconforming_count']}")
    print(f"Flagged records updated to 'invalid_target': {report['flagged_ids']}")
