from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Protocol

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ReturnDocument

from backend.utils.helpers import utcnow


@dataclass(frozen=True)
class RefreshTokenRecord:
    token_id: str
    user_id: str
    session_id: str
    family_id: str
    token_hash: str
    expires_at: object
    created_at: object
    updated_at: object
    revoked_at: Optional[object] = None
    rotated_at: Optional[object] = None
    replaced_by_token_id: Optional[str] = None
    reuse_detected: bool = False


class RefreshTokenRepository(Protocol):
    async def ensure_indexes(self) -> None: ...
    async def create_token(self, record: RefreshTokenRecord) -> RefreshTokenRecord: ...
    async def get_token(self, token_id: str) -> Optional[RefreshTokenRecord]: ...
    async def rotate_token(self, *, token_id: str, replacement_token_id: str) -> Optional[RefreshTokenRecord]: ...
    async def revoke_token(self, *, token_id: str, reuse_detected: bool = False) -> Optional[RefreshTokenRecord]: ...
    async def revoke_family(self, *, family_id: str) -> None: ...


def _doc_to_record(doc: Dict[str, object]) -> RefreshTokenRecord:
    now = utcnow()
    return RefreshTokenRecord(
        token_id=str(doc.get("token_id", "")),
        user_id=str(doc.get("user_id", "")),
        session_id=str(doc.get("session_id", "")),
        family_id=str(doc.get("family_id", "")),
        token_hash=str(doc.get("token_hash", "")),
        expires_at=doc.get("expires_at") or now,
        created_at=doc.get("created_at") or now,
        updated_at=doc.get("updated_at") or now,
        revoked_at=doc.get("revoked_at"),
        rotated_at=doc.get("rotated_at"),
        replaced_by_token_id=doc.get("replaced_by_token_id"),
        reuse_detected=bool(doc.get("reuse_detected", False)),
    )


class MongoRefreshTokenRepository:
    def __init__(self, db: AsyncIOMotorDatabase):
        self._col = db["refresh_tokens"]
        self._indexes_ready = False

    async def ensure_indexes(self) -> None:
        if self._indexes_ready:
            return
        await self._col.create_index("token_id", unique=True)
        await self._col.create_index([("user_id", 1), ("expires_at", -1)])
        await self._col.create_index([("family_id", 1), ("created_at", -1)])
        self._indexes_ready = True

    async def create_token(self, record: RefreshTokenRecord) -> RefreshTokenRecord:
        await self.ensure_indexes()
        await self._col.insert_one(
            {
                "token_id": record.token_id,
                "user_id": record.user_id,
                "session_id": record.session_id,
                "family_id": record.family_id,
                "token_hash": record.token_hash,
                "expires_at": record.expires_at,
                "created_at": record.created_at,
                "updated_at": record.updated_at,
                "revoked_at": record.revoked_at,
                "rotated_at": record.rotated_at,
                "replaced_by_token_id": record.replaced_by_token_id,
                "reuse_detected": record.reuse_detected,
            }
        )
        return record

    async def get_token(self, token_id: str) -> Optional[RefreshTokenRecord]:
        await self.ensure_indexes()
        doc = await self._col.find_one({"token_id": token_id})
        return _doc_to_record(doc) if doc else None

    async def rotate_token(self, *, token_id: str, replacement_token_id: str) -> Optional[RefreshTokenRecord]:
        await self.ensure_indexes()
        now = utcnow()
        doc = await self._col.find_one_and_update(
            {"token_id": token_id},
            {
                "$set": {
                    "rotated_at": now,
                    "revoked_at": now,
                    "updated_at": now,
                    "replaced_by_token_id": replacement_token_id,
                }
            },
            return_document=ReturnDocument.AFTER,
        )
        return _doc_to_record(doc) if doc else None

    async def revoke_token(self, *, token_id: str, reuse_detected: bool = False) -> Optional[RefreshTokenRecord]:
        await self.ensure_indexes()
        now = utcnow()
        doc = await self._col.find_one_and_update(
            {"token_id": token_id},
            {"$set": {"revoked_at": now, "updated_at": now, "reuse_detected": reuse_detected}},
            return_document=ReturnDocument.AFTER,
        )
        return _doc_to_record(doc) if doc else None

    async def revoke_family(self, *, family_id: str) -> None:
        await self.ensure_indexes()
        now = utcnow()
        await self._col.update_many(
            {"family_id": family_id, "revoked_at": None},
            {"$set": {"revoked_at": now, "updated_at": now, "reuse_detected": True}},
        )


class InMemoryRefreshTokenRepository:
    def __init__(self):
        self._tokens: Dict[str, RefreshTokenRecord] = {}

    async def ensure_indexes(self) -> None:
        return

    async def create_token(self, record: RefreshTokenRecord) -> RefreshTokenRecord:
        self._tokens[record.token_id] = record
        return record

    async def get_token(self, token_id: str) -> Optional[RefreshTokenRecord]:
        return self._tokens.get(token_id)

    async def rotate_token(self, *, token_id: str, replacement_token_id: str) -> Optional[RefreshTokenRecord]:
        existing = self._tokens.get(token_id)
        if not existing:
            return None
        now = utcnow()
        updated = RefreshTokenRecord(
            token_id=existing.token_id,
            user_id=existing.user_id,
            session_id=existing.session_id,
            family_id=existing.family_id,
            token_hash=existing.token_hash,
            expires_at=existing.expires_at,
            created_at=existing.created_at,
            updated_at=now,
            revoked_at=now,
            rotated_at=now,
            replaced_by_token_id=replacement_token_id,
            reuse_detected=existing.reuse_detected,
        )
        self._tokens[token_id] = updated
        return updated

    async def revoke_token(self, *, token_id: str, reuse_detected: bool = False) -> Optional[RefreshTokenRecord]:
        existing = self._tokens.get(token_id)
        if not existing:
            return None
        now = utcnow()
        updated = RefreshTokenRecord(
            token_id=existing.token_id,
            user_id=existing.user_id,
            session_id=existing.session_id,
            family_id=existing.family_id,
            token_hash=existing.token_hash,
            expires_at=existing.expires_at,
            created_at=existing.created_at,
            updated_at=now,
            revoked_at=now,
            rotated_at=existing.rotated_at,
            replaced_by_token_id=existing.replaced_by_token_id,
            reuse_detected=reuse_detected,
        )
        self._tokens[token_id] = updated
        return updated

    async def revoke_family(self, *, family_id: str) -> None:
        now = utcnow()
        for token_id, record in list(self._tokens.items()):
            if record.family_id != family_id or record.revoked_at is not None:
                continue
            self._tokens[token_id] = RefreshTokenRecord(
                token_id=record.token_id,
                user_id=record.user_id,
                session_id=record.session_id,
                family_id=record.family_id,
                token_hash=record.token_hash,
                expires_at=record.expires_at,
                created_at=record.created_at,
                updated_at=now,
                revoked_at=now,
                rotated_at=record.rotated_at,
                replaced_by_token_id=record.replaced_by_token_id,
                reuse_detected=True,
            )
