"""
AQQIP Stub Service — Phase 1 Placeholder

Keeps the container alive while infrastructure is being validated.
Each application service runs this stub until its implementation phase.
No business logic. Just a heartbeat.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)


def _handle_signal(signum: int, _frame: object) -> None:
    """Graceful shutdown on SIGTERM/SIGINT."""
    logging.info("Received signal %d — shutting down stub service.", signum)
    sys.exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="AQQIP stub service")
    parser.add_argument(
        "--name",
        type=str,
        default="unknown_service",
        help="Name of the service this stub represents",
    )
    args = parser.parse_args()

    logger = logging.getLogger(args.name)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    logger.info(
        "AQQIP stub service '%s' started. "
        "Awaiting implementation in a later phase.",
        args.name,
    )

    while True:
        time.sleep(60)
        logger.debug("Heartbeat — %s still alive.", args.name)


if __name__ == "__main__":
    main()
