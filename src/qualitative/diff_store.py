from typing import Dict, Any
from src.database.mongo_client import get_mongo_client
from config.settings import get_settings

def persist_semantic_diff(diff_record: dict) -> None:
    """
    Persists the qualitative signal (diff) to MongoDB.
    """
    settings = get_settings()
    client = get_mongo_client()
    db = client[settings.mongo.database]
    collection = db["semantic_diffs"]
    
    # Upsert based on unique index (ticker, section, year_from, year_to)
    filter_query = {
        "ticker": diff_record["ticker"],
        "section": diff_record["section"],
        "year_from": diff_record["year_from"],
        "year_to": diff_record["year_to"]
    }
    
    collection.update_one(filter_query, {"$set": diff_record}, upsert=True)
