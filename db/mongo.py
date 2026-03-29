from __future__ import annotations

from typing import Optional
from urllib.parse import urlparse

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import MongoClient
from pymongo.database import Database

from backend.config.settings import settings
from backend.utils.logger import get_logger

logger = get_logger("mongo")

_async_client: Optional[AsyncIOMotorClient] = None
_async_db: Optional[AsyncIOMotorDatabase] = None
_sync_client: Optional[MongoClient] = None
_sync_db: Optional[Database] = None


def _db_name_from_uri(uri: str) -> str:
    parsed = urlparse(uri)
    path = (parsed.path or "").lstrip("/")
    return path or "revops_ai"


def get_async_client() -> AsyncIOMotorClient:
    global _async_client
    if _async_client is None:
        _async_client = AsyncIOMotorClient(
            settings.mongodb_uri,
            serverSelectionTimeoutMS=settings.mongo_server_selection_timeout_ms,
            connectTimeoutMS=settings.mongo_connect_timeout_ms,
            socketTimeoutMS=settings.mongo_socket_timeout_ms,
        )
    return _async_client


def get_database() -> AsyncIOMotorDatabase:
    global _async_db
    if _async_db is None:
        _async_db = get_async_client()[_db_name_from_uri(settings.mongodb_uri)]
        logger.info("mongo_async_ready", database=_async_db.name)
    return _async_db


def get_sync_client() -> MongoClient:
    global _sync_client
    if _sync_client is None:
        _sync_client = MongoClient(
            settings.mongodb_uri,
            serverSelectionTimeoutMS=settings.mongo_server_selection_timeout_ms,
            connectTimeoutMS=settings.mongo_connect_timeout_ms,
            socketTimeoutMS=settings.mongo_socket_timeout_ms,
        )
    return _sync_client


def get_sync_database() -> Database:
    global _sync_db
    if _sync_db is None:
        _sync_db = get_sync_client()[_db_name_from_uri(settings.mongodb_uri)]
        logger.info("mongo_sync_ready", database=_sync_db.name)
    return _sync_db


def close_clients() -> None:
    global _async_client, _async_db, _sync_client, _sync_db
    if _async_client is not None:
        _async_client.close()
    if _sync_client is not None:
        _sync_client.close()
    _async_client = None
    _async_db = None
    _sync_client = None
    _sync_db = None
