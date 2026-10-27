"""
Pytest configuration for AQQIP test suite.

Provides shared fixtures for database connections and configuration.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Generator

import pytest

# When running tests on the host machine (outside Docker), the internal Docker 
# DNS names ('mysql', 'mongodb') are unreachable.
# We override them to 'localhost' for testing unless explicitly overridden.
if not Path("/.dockerenv").exists() and not os.environ.get("RUNNING_IN_DOCKER"):
    os.environ["MYSQL_HOST"] = "localhost"
    os.environ["MONGO_HOST"] = "localhost"

# Ensure project root is on Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def settings():
    """Load AQQIP settings from environment / .env file."""
    from config.settings import get_settings
    return get_settings()


@pytest.fixture(scope="session")
def mysql_connection(settings):
    """
    Provide a MySQL connection for integration tests.

    Skips if MySQL is unreachable (allows running unit tests without Docker).
    """
    import mysql.connector
    try:
        conn = mysql.connector.connect(
            host=os.getenv("MYSQL_HOST", settings.mysql.host),
            port=int(os.getenv("MYSQL_PORT", settings.mysql.port)),
            database=settings.mysql.database,
            user=settings.mysql.user,
            password=settings.mysql.password,
            connect_timeout=10,
        )
        yield conn
        conn.close()
    except Exception as e:
        pytest.skip(f"MySQL not reachable: {e}")


@pytest.fixture(scope="session")
def mongo_client(settings):
    """
    Provide a MongoDB client for integration tests.

    Skips if MongoDB is unreachable.
    """
    from pymongo import MongoClient
    from pymongo.errors import ServerSelectionTimeoutError

    try:
        client = MongoClient(
            host=os.getenv("MONGO_HOST", settings.mongo.host),
            port=int(os.getenv("MONGO_PORT", settings.mongo.port)),
            serverSelectionTimeoutMS=5000,
        )
        # Force connection attempt
        client.admin.command("ping")
        yield client
        client.close()
    except ServerSelectionTimeoutError as e:
        pytest.skip(f"MongoDB not reachable: {e}")


@pytest.fixture(scope="session")
def mongo_db(mongo_client, settings):
    """Return the AQQIP MongoDB database."""
    return mongo_client[settings.mongo.database]
