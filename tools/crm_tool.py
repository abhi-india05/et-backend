"""CRM tool — thread-safe access to MongoDB CRM data with strict multi-tenancy."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from pymongo.collection import Collection

from backend.db.mongo import get_sync_database
from backend.utils.helpers import days_since, now_iso
from backend.utils.logger import get_logger

logger = get_logger("crm_tool")


def _get_accounts_col() -> Collection:
    return get_sync_database()["accounts"]


def _get_usage_col() -> Collection:
    return get_sync_database()["usage_data"]


def get_all_accounts(user_id: str) -> List[Dict[str, Any]]:
    if not user_id:
        raise ValueError("user_id is required")
    cursor = _get_accounts_col().find({"user_id": user_id}, {"_id": 0})
    return list(cursor)


def get_account_by_id(account_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    if not user_id:
        raise ValueError("user_id is required")
    return _get_accounts_col().find_one({"account_id": account_id, "user_id": user_id}, {"_id": 0})


def get_accounts_by_stage(stage: str, user_id: str) -> List[Dict[str, Any]]:
    if not user_id:
        raise ValueError("user_id is required")
    # Case-insensitive stage match
    cursor = _get_accounts_col().find(
        {"stage": {"$regex": f"^{stage}$", "$options": "i"}, "user_id": user_id},
        {"_id": 0}
    )
    return list(cursor)


def get_at_risk_deals(inactivity_days: int = 10, *, user_id: str) -> List[Dict[str, Any]]:
    if not user_id:
        raise ValueError("user_id is required")
        
    all_accounts = get_all_accounts(user_id)
    at_risk: List[Dict[str, Any]] = []
    
    for acc in all_accounts:
        if acc.get("stage") in ["Closed Won", "Closed Lost"]:
            continue
        days = days_since(acc.get("last_activity", ""))
        if days >= inactivity_days:
            acc["days_inactive"] = days
            at_risk.append(acc)
            
    return sorted(at_risk, key=lambda x: x.get("days_inactive", 0), reverse=True)


def get_usage_data(account_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    if not user_id:
        raise ValueError("user_id is required")
    return _get_usage_col().find_one({"account_id": account_id, "user_id": user_id}, {"_id": 0})


def get_all_usage_data(user_id: str) -> List[Dict[str, Any]]:
    if not user_id:
        raise ValueError("user_id is required")
    return list(_get_usage_col().find({"user_id": user_id}, {"_id": 0}))


def update_deal_stage(account_id: str, new_stage: str, notes: str = "", *, user_id: str) -> Dict[str, Any]:
    if not user_id:
        raise ValueError("user_id is required")
        
    patch = {
        "stage": new_stage,
        "last_updated": now_iso(),
        "notes": notes,
    }
    result = _get_accounts_col().update_one(
        {"account_id": account_id, "user_id": user_id},
        {"$set": patch}
    )
    
    logger.info("deal_stage_updated", account_id=account_id, stage=new_stage, user_id=user_id)
    return {
        "success": result.modified_count > 0, 
        "account_id": account_id, 
        "new_stage": new_stage, 
        "timestamp": now_iso()
    }


def log_activity(account_id: str, activity_type: str, description: str, *, user_id: str) -> Dict[str, Any]:
    if not user_id:
        raise ValueError("user_id is required")
        
    patch = {
        "last_activity": now_iso(),
        "last_activity_type": activity_type,
        "last_activity_desc": description,
    }
    result = _get_accounts_col().update_one(
        {"account_id": account_id, "user_id": user_id},
        {"$set": patch}
    )
    
    logger.info("activity_logged", account_id=account_id, type=activity_type, user_id=user_id)
    return {
        "success": result.modified_count > 0, 
        "account_id": account_id, 
        "activity_type": activity_type, 
        "timestamp": now_iso()
    }


def add_new_lead(lead_data: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    if not user_id:
        raise ValueError("user_id is required")
        
    col = _get_accounts_col()
    count = col.count_documents({"user_id": user_id})
    
    new_account = {
        "user_id": user_id,
        "account_id": f"acc_{count + 100:03d}",
        "company": lead_data.get("company", "Unknown"),
        "contact_name": lead_data.get("name", "Unknown"),
        "email": lead_data.get("email", ""),
        "deal_value": lead_data.get("deal_value", 0),
        "stage": "Prospecting",
        "last_activity": now_iso(),
        "days_in_stage": 0,
        "arr": 0,
        "health_score": 50,
        "open_tickets": 0,
        "logins_last_30_days": 0,
        "nps_score": 0,
        "industry": lead_data.get("industry", "Unknown"),
        "employee_count": lead_data.get("employee_count", 0),
        "created_at": now_iso(),
        "updated_at": now_iso()
    }
    
    col.insert_one(new_account.copy())
    logger.info("lead_added", account_id=new_account["account_id"], user_id=user_id)
    
    # Remove _id before returning
    new_account.pop("_id", None)
    return new_account


def get_pipeline_stats(user_id: str) -> Dict[str, Any]:
    if not user_id:
        raise ValueError("user_id is required")
        
    accounts = get_all_accounts(user_id)
    stages: Dict[str, Dict[str, Any]] = {}
    total_value = 0.0
    
    for acc in accounts:
        stage = acc.get("stage", "Unknown")
        if stage not in stages:
            stages[stage] = {"count": 0, "total_value": 0.0}
        stages[stage]["count"] += 1
        val = acc.get("deal_value", 0)
        stages[stage]["total_value"] += val
        if stage != "Closed Lost":
            total_value += val
            
    return {
        "stages": stages,
        "total_pipeline_value": total_value,
        "total_accounts": len(accounts),
        "timestamp": now_iso(),
    }


def search_accounts(query: str, user_id: str) -> List[Dict[str, Any]]:
    if not user_id:
        raise ValueError("user_id is required")
        
    q = query.lower()
    return [
        a
        for a in get_all_accounts(user_id)
        if q in a.get("company", "").lower()
        or q in a.get("contact_name", "").lower()
        or q in a.get("industry", "").lower()
    ]