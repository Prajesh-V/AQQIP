"""
AQQIP MongoDB Client

Provides a singleton MongoDB client and database accessor.
"""
from __future__ import annotations

from pymongo import MongoClient
from pymongo.database import Database

from config.settings import get_settings

_client: MongoClient | None = None


def get_mongo_client() -> MongoClient:
    """Return the global MongoClient instance."""
    global _client
    if _client is None:
        settings = get_settings()
        _client = MongoClient(settings.mongo.uri)
    return _client


def get_mongo_db() -> Database:
    """Return the configured AQQIP raw database."""
    settings = get_settings()
    client = get_mongo_client()
    return client[settings.mongo.database]
