"""
AQQIP MySQL Client

Provides a context manager for MySQL connections.
"""
from __future__ import annotations

import contextlib
from typing import Generator

import mysql.connector
from mysql.connector.connection import MySQLConnection

from config.settings import get_settings


@contextlib.contextmanager
def get_mysql_connection() -> Generator[MySQLConnection, None, None]:
    """Provide a managed MySQL connection. Commits on exit if no exception, otherwise rolls back."""
    settings = get_settings()
    conn = None
    try:
        conn = mysql.connector.connect(
            host=settings.mysql.host,
            port=settings.mysql.port,
            database=settings.mysql.database,
            user=settings.mysql.user,
            password=settings.mysql.password,
            autocommit=False
        )
        yield conn
        conn.commit()
    except Exception:
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()
