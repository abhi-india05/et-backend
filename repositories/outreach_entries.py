from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Tuple

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ReturnDocument

from backend.models.schemas import OutreachEntry, OutreachEntryStatus
from backend.utils.helpers import utcnow


class OutreachEntryRepository(Protocol):
    async def ensure_indexes(self) -> None: ...
    async def create_entry(self, entry: OutreachEntry) -> OutreachEntry: ...
    async def get_entry(self, entry_id: str) -> Optional[OutreachEntry]: ...
    async def list_entries(
        self,
        *,
        user_id: str,
        page: int,
        page_size: int,
        status: Optional[str] = None,
        company: Optional[str] = None,
    ) -> Tuple[List[OutreachEntry], int]: ...
    async def update_status(self, entry_id: str, status: OutreachEntryStatus) -> Optional[OutreachEntry]: ...


def _doc_to_entry(doc: Dict[str, Any]) -> OutreachEntry:
    now = utcnow()
    return OutreachEntry(
        id=str(doc.get("id", "")),
        user_id=str(doc.get("user_id", "")),
        company_name=str(doc.get("company_name", "")),
        company_domain=doc.get("company_domain"),
        outreach_type=str(doc.get("outreach_type", "email")),
        message=doc.get("message"),
        status=OutreachEntryStatus(doc.get("status", "draft")),
        created_at=doc.get("created_at") or now,
        updated_at=doc.get("updated_at") or now,
    )


class MongoOutreachEntryRepository:
    def __init__(self, db: AsyncIOMotorDatabase):
        self._col = db["outreach_entries"]
        self._indexes_ready = False

    async def ensure_indexes(self) -> None:
        if self._indexes_ready:
            return
        await self._col.create_index("id", unique=True)
        await self._col.create_index([("user_id", 1), ("created_at", -1)])
        await self._col.create_index([("company_name", 1)])
        self._indexes_ready = True

    async def create_entry(self, entry: OutreachEntry) -> OutreachEntry:
        await self.ensure_indexes()
        doc = entry.model_dump()
        await self._col.insert_one(doc)
        return entry

    async def get_entry(self, entry_id: str) -> Optional[OutreachEntry]:
        doc = await self._col.find_one({"id": entry_id})
        return _doc_to_entry(doc) if doc else None

    async def list_entries(
        self,
        *,
        user_id: str,
        page: int,
        page_size: int,
        status: Optional[str] = None,
        company: Optional[str] = None,
    ) -> Tuple[List[OutreachEntry], int]:
        await self.ensure_indexes()
        filters: Dict[str, Any] = {"user_id": user_id}
        if status:
            filters["status"] = status
        if company:
            filters["company_name"] = company
            
        total = await self._col.count_documents(filters)
        cursor = (
            self._col.find(filters, {"_id": 0})
            .sort("created_at", -1)
            .skip(max(0, (page - 1) * page_size))
            .limit(page_size)
        )
        items: List[OutreachEntry] = []
        async for doc in cursor:
            items.append(_doc_to_entry(doc))
        return items, total

    async def update_status(self, entry_id: str, status: OutreachEntryStatus) -> Optional[OutreachEntry]:
        await self.ensure_indexes()
        now = utcnow()
        patch: Dict[str, Any] = {"status": status.value, "updated_at": now}
        
        doc = await self._col.find_one_and_update(
            {"id": entry_id},
            {"$set": patch},
            return_document=ReturnDocument.AFTER,
        )
        return _doc_to_entry(doc) if doc else None
