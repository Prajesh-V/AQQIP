"""
AQQIP Ingestion Scheduler

Orchestrates periodic jobs (e.g., EDGAR near-real-time monitoring).
"""
from __future__ import annotations

import time
from apscheduler.schedulers.background import BackgroundScheduler

from src.ingestion.edgar_monitor import EdgarMonitor
from src.utils.logging import get_logger

logger = get_logger("scheduler")


def start_scheduler() -> BackgroundScheduler:
    """Start the background scheduler for ingestion jobs."""
    scheduler = BackgroundScheduler()
    monitor = EdgarMonitor()

    # Poll SEC EDGAR every 2 minutes. This is conservative and stays
    # well under the 10 requests/second fair-access limit.
    scheduler.add_job(
        monitor.run_polling_cycle,
        "interval",
        minutes=2,
        id="edgar_monitor_job",
        replace_existing=True
    )

    scheduler.start()
    logger.info("Ingestion scheduler started. EDGAR polling set to 2-minute interval.")
    return scheduler


if __name__ == "__main__":
    # If run directly as a daemon
    _scheduler = start_scheduler()
    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down scheduler...")
        _scheduler.shutdown()
