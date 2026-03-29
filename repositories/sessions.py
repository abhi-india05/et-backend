from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ReturnDocument

from backend.utils.helpers import parse_date, utcnow


@dataclass(frozen=True)
class SessionInDB:
    session_id: str
    owner_user_id: str
    task_type: str
    status: str
    input_data: Dict[str, Any]
    plan: Dict[str, Any]
    request_id: Optional[str]
    created_at: object
    updated_at: object
    completed_at: Optional[object] = None
    error: Optional[str] = None
    final_output: Dict[str, Any] = field(default_factory=dict)


class SessionRepository(Protocol):
    async def ensure_indexes(self) -> None: ...
    async def get_session(self, *, session_id: str) -> Optional[SessionInDB]: ...
    async def create_session(
        self,
        *,
        session_id: str,
        owner_user_id: str,
        task_type: str,
        input_data: Dict[str, Any],
        plan: Dict[str, Any],
        request_id: Optional[str],
        status: str = "pending",
    ) -> SessionInDB: ...
    async def update_session(
        self,
        *,
        session_id: str,
        status: str,
        plan: Optional[Dict[str, Any]] = None,
        final_output: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> Optional[SessionInDB]: ...
    async def list_sessions(
        self,
        *,
        owner_user_id: str,
        page: int,
        page_size: int,
        status: Optional[str] = None,
        task_type: Optional[str] = None,
        created_from: Optional[str] = None,
        created_to: Optional[str] = None,
    ) -> Tuple[List[SessionInDB], int, int]: ...


def _doc_to_session(doc: Dict[str, Any]) -> SessionInDB:
    now = utcnow()
    return SessionInDB(
        session_id=str(doc.get("session_id", "")),
        owner_user_id=str(doc.get("owner_user_id", "")),
        task_type=str(doc.get("task_type", "")),
        status=str(doc.get("status", "pending")),
        input_data=dict(doc.get("input_data", {})),
        plan=dict(doc.get("plan", {})),
        request_id=doc.get("request_id"),
        created_at=doc.get("created_at") or now,
        updated_at=doc.get("updated_at") or now,
        completed_at=doc.get("completed_at"),
        error=doc.get("error"),
        final_output=dict(doc.get("final_output", {})),
    )


class MongoSessionRepository:
    def __init__(self, db: AsyncIOMotorDatabase):
        self._col = db["sessions"]
        self._indexes_ready = False

    async def ensure_indexes(self) -> None:
        if self._indexes_ready:
            return
        await self._col.create_index("session_id", unique=True)
        await self._col.create_index([("status", 1), ("created_at", -1)])
        await self._col.create_index([("owner_user_id", 1), ("created_at", -1)])
        self._indexes_ready = True

    async def create_session(
        self,
        *,
        session_id: str,
        owner_user_id: str,
        task_type: str,
        input_data: Dict[str, Any],
        plan: Dict[str, Any],
        request_id: Optional[str],
        status: str = "pending",
    ) -> SessionInDB:
        await self.ensure_indexes()
        now = utcnow()
        doc = {
            "session_id": session_id,
            "owner_user_id": owner_user_id,
            "task_type": task_type,
            "status": status,
            "input_data": input_data,
            "plan": plan,
            "request_id": request_id,
            "created_at": now,
            "updated_at": now,
            "completed_at": None,
            "error": None,
            "final_output": {},
        }
        await self._col.insert_one(doc)
        return _doc_to_session(doc)

    async def get_session(self, *, session_id: str) -> Optional[SessionInDB]:
        await self.ensure_indexes()
        doc = await self._col.find_one({"session_id": session_id}, {"_id": 0})
        return _doc_to_session(doc) if doc else None

    async def update_session(
        self,
        *,
        session_id: str,
        status: str,
        plan: Optional[Dict[str, Any]] = None,
        final_output: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> Optional[SessionInDB]:
        await self.ensure_indexes()
        now = utcnow()
        patch: Dict[str, Any] = {"status": status, "updated_at": now}
        if plan is not None:
            patch["plan"] = plan
        if final_output is not None:
            patch["final_output"] = final_output
        if error is not None:
            patch["error"] = error
        if status in {"completed", "completed_with_errors", "failed"}:
            patch["completed_at"] = now
        doc = await self._col.find_one_and_update(
            {"session_id": session_id},
            {"$set": patch},
            return_document=ReturnDocument.AFTER,
        )
        return _doc_to_session(doc) if doc else None

    async def list_sessions(
        self,
        *,
        owner_user_id: str,
        page: int,
        page_size: int,
        status: Optional[str] = None,
        task_type: Optional[str] = None,
        created_from: Optional[str] = None,
        created_to: Optional[str] = None,
    ) -> Tuple[List[SessionInDB], int, int]:
        await self.ensure_indexes()
        filters: Dict[str, Any] = {"owner_user_id": owner_user_id}
        if status:
            filters["status"] = status
        if task_type:
            filters["task_type"] = task_type
        from_dt = parse_date(created_from or "")
        to_dt = parse_date(created_to or "")
        created_at: Dict[str, Any] = {}
        if from_dt:
            created_at["$gte"] = from_dt
        if to_dt:
            created_at["$lte"] = to_dt
        if created_at:
            filters["created_at"] = created_at
        total = await self._col.count_documents(filters)
        running_filters: Dict[str, Any] = {"owner_user_id": owner_user_id, "status": "running"}
        if task_type:
            running_filters["task_type"] = task_type
        running = await self._col.count_documents(running_filters)
        cursor = (
            self._col.find(filters, {"_id": 0})
            .sort("created_at", -1)
            .skip(max(0, (page - 1) * page_size))
            .limit(page_size)
        )
        items: List[SessionInDB] = []
        async for doc in cursor:
            items.append(_doc_to_session(doc))
        return items, total, running


class InMemorySessionRepository:
    def __init__(self):
        self._sessions: Dict[str, SessionInDB] = {}

    async def ensure_indexes(self) -> None:
        return

    async def create_session(
        self,
        *,
        session_id: str,
        owner_user_id: str,
        task_type: str,
        input_data: Dict[str, Any],
        plan: Dict[str, Any],
        request_id: Optional[str],
        status: str = "pending",
    ) -> SessionInDB:
        now = utcnow()
        session = SessionInDB(
            session_id=session_id,
            owner_user_id=owner_user_id,
            task_type=task_type,
            status=status,
            input_data=dict(input_data),
            plan=dict(plan),
            request_id=request_id,
            created_at=now,
            updated_at=now,
            completed_at=None,
            error=None,
            final_output={},
        )
        self._sessions[session_id] = session
        return session

    async def get_session(self, *, session_id: str) -> Optional[SessionInDB]:
        return self._sessions.get(session_id)

    async def update_session(
        self,
        *,
        session_id: str,
        status: str,
        plan: Optional[Dict[str, Any]] = None,
        final_output: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> Optional[SessionInDB]:
        existing = self._sessions.get(session_id)
        if not existing:
            return None
        now = utcnow()
        updated = SessionInDB(
            session_id=existing.session_id,
            owner_user_id=existing.owner_user_id,
            task_type=existing.task_type,
            status=status,
            input_data=existing.input_data,
            plan=plan if plan is not None else existing.plan,
            request_id=existing.request_id,
            created_at=existing.created_at,
            updated_at=now,
            completed_at=now if status in {"completed", "completed_with_errors", "failed"} else existing.completed_at,
            error=error if error is not None else existing.error,
            final_output=final_output if final_output is not None else existing.final_output,
        )
        self._sessions[session_id] = updated
        return updated

    async def list_sessions(
        self,
        *,
        owner_user_id: str,
        page: int,
        page_size: int,
        status: Optional[str] = None,
        task_type: Optional[str] = None,
        created_from: Optional[str] = None,
        created_to: Optional[str] = None,
    ) -> Tuple[List[SessionInDB], int, int]:
        from_dt = parse_date(created_from or "")
        to_dt = parse_date(created_to or "")
        items = [session for session in self._sessions.values() if session.owner_user_id == owner_user_id]
        if status:
            items = [session for session in items if session.status == status]
        if task_type:
            items = [session for session in items if session.task_type == task_type]
        if from_dt:
            items = [session for session in items if session.created_at >= from_dt]
        if to_dt:
            items = [session for session in items if session.created_at <= to_dt]
        items.sort(key=lambda session: session.created_at, reverse=True)
        total = len(items)
        running = len(
            [
                session
                for session in self._sessions.values()
                if session.owner_user_id == owner_user_id
                and session.status == "running"
                and (not task_type or session.task_type == task_type)
            ]
        )
        start = max(0, (page - 1) * page_size)
        end = start + page_size
        return items[start:end], total, running
