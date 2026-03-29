from __future__ import annotations

from functools import lru_cache

from backend.db.mongo import get_database
from backend.repositories.refresh_tokens import MongoRefreshTokenRepository, RefreshTokenRepository
from backend.repositories.sessions import MongoSessionRepository, SessionRepository
from backend.repositories.users import MongoUserRepository, UserRepository
from backend.repositories.customers import MongoCustomerRepository, CustomerRepository
from backend.repositories.outreach_entries import MongoOutreachEntryRepository, OutreachEntryRepository


@lru_cache(maxsize=1)
def get_user_repo() -> UserRepository:
    return MongoUserRepository(get_database())


@lru_cache(maxsize=1)


@lru_cache(maxsize=1)
def get_session_repo() -> SessionRepository:
    return MongoSessionRepository(get_database())


@lru_cache(maxsize=1)
def get_refresh_token_repo() -> RefreshTokenRepository:
    return MongoRefreshTokenRepository(get_database())


@lru_cache(maxsize=1)
def get_outreach_entry_repo() -> OutreachEntryRepository:
    return MongoOutreachEntryRepository(get_database())


@lru_cache(maxsize=1)
def get_customer_repo() -> CustomerRepository:
    return MongoCustomerRepository(get_database())
