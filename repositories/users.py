from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Protocol

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ReturnDocument

from backend.utils.helpers import utcnow


@dataclass(frozen=True)
class UserInDB:
    user_id: str
    username: str
    password_hash: str
    role: str
    created_at: object
    updated_at: object
    last_login_at: Optional[object] = None


class UserRepository(Protocol):
    async def ensure_indexes(self) -> None: ...
    async def create_user(self, *, username: str, password_hash: str, role: str = "user") -> UserInDB: ...
    async def get_by_username(self, username: str) -> Optional[UserInDB]: ...
    async def get_by_id(self, user_id: str) -> Optional[UserInDB]: ...
    async def update_last_login(self, user_id: str) -> Optional[UserInDB]: ...
    async def ensure_admin_user(self, *, username: str, password_hash: str) -> Optional[UserInDB]: ...


def _doc_to_user(doc: Dict[str, object]) -> UserInDB:
    now = utcnow()
    return UserInDB(
        user_id=str(doc["_id"]),
        username=str(doc.get("username", "")),
        password_hash=str(doc.get("password_hash", "")),
        role=str(doc.get("role", "user")),
        created_at=doc.get("created_at") or now,
        updated_at=doc.get("updated_at") or now,
        last_login_at=doc.get("last_login_at"),
    )


class MongoUserRepository:
    def __init__(self, db: AsyncIOMotorDatabase):
        self._col = db["users"]
        self._indexes_ready = False

    async def ensure_indexes(self) -> None:
        if self._indexes_ready:
            return
        await self._col.create_index("username", unique=True)
        await self._col.create_index([("role", 1), ("created_at", -1)])
        self._indexes_ready = True

    async def create_user(self, *, username: str, password_hash: str, role: str = "user") -> UserInDB:
        await self.ensure_indexes()
        now = utcnow()
        doc = {
            "username": username,
            "password_hash": password_hash,
            "role": role,
            "created_at": now,
            "updated_at": now,
            "last_login_at": None,
        }
        result = await self._col.insert_one(doc)
        doc["_id"] = result.inserted_id
        return _doc_to_user(doc)

    async def get_by_username(self, username: str) -> Optional[UserInDB]:
        await self.ensure_indexes()
        doc = await self._col.find_one({"username": username})
        return _doc_to_user(doc) if doc else None

    async def get_by_id(self, user_id: str) -> Optional[UserInDB]:
        await self.ensure_indexes()
        try:
            oid = ObjectId(user_id)
        except Exception:
            return None
        doc = await self._col.find_one({"_id": oid})
        return _doc_to_user(doc) if doc else None

    async def update_last_login(self, user_id: str) -> Optional[UserInDB]:
        await self.ensure_indexes()
        try:
            oid = ObjectId(user_id)
        except Exception:
            return None
        now = utcnow()
        doc = await self._col.find_one_and_update(
            {"_id": oid},
            {"$set": {"last_login_at": now, "updated_at": now}},
            return_document=ReturnDocument.AFTER,
        )
        return _doc_to_user(doc) if doc else None

    async def ensure_admin_user(self, *, username: str, password_hash: str) -> Optional[UserInDB]:
        await self.ensure_indexes()
        existing = await self.get_by_username(username)
        if existing:
            return existing
        return await self.create_user(username=username, password_hash=password_hash, role="admin")


class InMemoryUserRepository:
    def __init__(self):
        self._users_by_id: Dict[str, UserInDB] = {}
        self._users_by_username: Dict[str, UserInDB] = {}
        self._next = 1

    async def ensure_indexes(self) -> None:
        return

    async def create_user(self, *, username: str, password_hash: str, role: str = "user") -> UserInDB:
        if username in self._users_by_username:
            raise ValueError("Username already exists")
        user_id = str(self._next)
        self._next += 1
        now = utcnow()
        user = UserInDB(
            user_id=user_id,
            username=username,
            password_hash=password_hash,
            role=role,
            created_at=now,
            updated_at=now,
            last_login_at=None,
        )
        self._users_by_id[user_id] = user
        self._users_by_username[username] = user
        return user

    async def get_by_username(self, username: str) -> Optional[UserInDB]:
        return self._users_by_username.get(username)

    async def get_by_id(self, user_id: str) -> Optional[UserInDB]:
        return self._users_by_id.get(user_id)

    async def update_last_login(self, user_id: str) -> Optional[UserInDB]:
        existing = self._users_by_id.get(user_id)
        if not existing:
            return None
        now = utcnow()
        updated = UserInDB(
            user_id=existing.user_id,
            username=existing.username,
            password_hash=existing.password_hash,
            role=existing.role,
            created_at=existing.created_at,
            updated_at=now,
            last_login_at=now,
        )
        self._users_by_id[user_id] = updated
        self._users_by_username[updated.username] = updated
        return updated

    async def ensure_admin_user(self, *, username: str, password_hash: str) -> Optional[UserInDB]:
        existing = self._users_by_username.get(username)
        if existing:
            return existing
        return await self.create_user(username=username, password_hash=password_hash, role="admin")
