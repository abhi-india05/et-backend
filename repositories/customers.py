from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, Tuple

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ReturnDocument

from backend.models.schemas import Customer, OutreachEntryStatus
from backend.utils.helpers import utcnow


class CustomerRepository(Protocol):
    async def ensure_indexes(self) -> None: ...
    async def create_customer(self, customer: Customer) -> Customer: ...
    async def get_customer(self, customer_id: str) -> Optional[Customer]: ...
    async def update_marked_as_customer_at(
        self,
        *,
        customer_id: str,
        marked_as_customer_at: datetime,
    ) -> Optional[Customer]: ...
    async def get_by_source_entry(self, *, user_id: str, source_entry_id: str) -> Optional[Customer]: ...
    async def list_customers(
        self,
        *,
        user_id: str,
        page: int,
        page_size: int,
        query: Optional[str] = None,
    ) -> Tuple[List[Customer], int]: ...


def _doc_to_customer(doc: Dict[str, Any]) -> Customer:
    raw_status = doc.get("source_outreach_status")
    source_status: Optional[OutreachEntryStatus] = None
    if raw_status:
        try:
            source_status = OutreachEntryStatus(str(raw_status))
        except ValueError:
            source_status = None

    return Customer(
        id=str(doc.get("id", "")),
        user_id=str(doc.get("user_id", "")),
        company_name=str(doc.get("company_name", "")),
        company_domain=doc.get("company_domain"),
        contact_name=doc.get("contact_name"),
        contact_email=doc.get("contact_email"),
        notes=doc.get("notes"),
        source_entry_id=doc.get("source_entry_id"),
        source_outreach_status=source_status,
        marked_as_customer_at=doc.get("marked_as_customer_at"),
        created_at=doc.get("created_at"),
        updated_at=doc.get("updated_at"),
    )


class MongoCustomerRepository:
    def __init__(self, db: AsyncIOMotorDatabase):
        self._col = db["customers"]
        self._indexes_ready = False

    async def ensure_indexes(self) -> None:
        if self._indexes_ready:
            return
        await self._col.create_index("id", unique=True)
        await self._col.create_index([("user_id", 1), ("created_at", -1)])
        await self._col.create_index([("user_id", 1), ("source_entry_id", 1)])
        await self._col.create_index([("user_id", 1), ("company_name", 1)])
        self._indexes_ready = True

    async def create_customer(self, customer: Customer) -> Customer:
        await self.ensure_indexes()
        doc = customer.model_dump()
        if isinstance(doc.get("source_outreach_status"), OutreachEntryStatus):
            doc["source_outreach_status"] = doc["source_outreach_status"].value
        await self._col.insert_one(doc)
        return customer

    async def get_customer(self, customer_id: str) -> Optional[Customer]:
        await self.ensure_indexes()
        doc = await self._col.find_one({"id": customer_id}, {"_id": 0})
        return _doc_to_customer(doc) if doc else None

    async def update_marked_as_customer_at(
        self,
        *,
        customer_id: str,
        marked_as_customer_at: datetime,
    ) -> Optional[Customer]:
        await self.ensure_indexes()
        doc = await self._col.find_one_and_update(
            {"id": customer_id},
            {"$set": {"marked_as_customer_at": marked_as_customer_at, "updated_at": utcnow()}},
            projection={"_id": 0},
            return_document=ReturnDocument.AFTER,
        )
        return _doc_to_customer(doc) if doc else None

    async def get_by_source_entry(self, *, user_id: str, source_entry_id: str) -> Optional[Customer]:
        await self.ensure_indexes()
        doc = await self._col.find_one(
            {"user_id": user_id, "source_entry_id": source_entry_id},
            {"_id": 0},
        )
        return _doc_to_customer(doc) if doc else None

    async def list_customers(
        self,
        *,
        user_id: str,
        page: int,
        page_size: int,
        query: Optional[str] = None,
    ) -> Tuple[List[Customer], int]:
        await self.ensure_indexes()
        filters: Dict[str, Any] = {"user_id": user_id}
        if query:
            search = str(query).strip()
            if search:
                filters["$or"] = [
                    {"company_name": {"$regex": search, "$options": "i"}},
                    {"contact_name": {"$regex": search, "$options": "i"}},
                    {"contact_email": {"$regex": search, "$options": "i"}},
                ]

        total = await self._col.count_documents(filters)
        cursor = (
            self._col.find(filters, {"_id": 0})
            .sort("created_at", -1)
            .skip(max(0, (page - 1) * page_size))
            .limit(page_size)
        )
        items: List[Customer] = []
        async for doc in cursor:
            items.append(_doc_to_customer(doc))
        return items, total
