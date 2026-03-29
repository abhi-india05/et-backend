# Production-Grade Backend Upgrade Plan

This document outlines the architectural changes required to finalize the production-grade upgrade of the FastAPI backend. It addresses all 14 points defined in the requirements.

## User Review Required

> [!WARNING]
> Moving routes to `api/routes/` and prefixing with `/api/v1/` will be a breaking change for any clients currently relying on the root endpoints (e.g., `/auth/login` instead of `/api/v1/auth/login`).
> The `AuthMeResponse` schema will be strictly aligned with the `AuthUser` model, standardizing on the `role` property.
> In-memory repositories (`InMemoryProductRepository`, `InMemoryUserRepository`) will be permanently removed to strictly isolate production code.

## Proposed Changes

---
### 1. API Routing Layer & Service Layer
Moving business logic out of `main.py` into proper services and grouping endpoints under routers with an `/api/v1` prefix.

#### [NEW] `backend/api/routes/auth.py`
Create router for `/auth/login`, `/auth/register`, `/auth/refresh`, `/auth/logout`, `/auth/me`. Adds rate limiting to register and refresh. Wraps responses in standard `{ data, error, meta }` format if needed, but per requirement, we will ensure consistent schema (if legacy frontend expects direct responses, we should discuss).
Wait, requirement 5 says:
> Ensure ALL responses follow consistent format:
> `{ "data": ..., "error": null, "meta": {...} }`
This requires rewriting response schemas to use a standard wrapper!

#### [NEW] `backend/api/routes/products.py`
Create router for `/products` endpoints, implementing `page`/`page_size` pagination, filtering by `name` and date range, and exposing `DELETE /products/{id}`.

#### [NEW] `backend/services/product_service.py`
A new service layer to handle product business logic (creating, listing, fetching, updating, soft-deleting). `ProductRepository` will only handle pure DB operations.

#### [MODIFY] `backend/main.py`
- Remove all route definitions.
- Import and include `api_router` from `backend/api/router.py` prefixing with `/api/v1/`.

---
### 2. Error Handling Standardization
Move `APIError` out of `main.py` into a shared module.

#### [NEW] `backend/utils/errors.py`
Defines `APIError` and standard error formats.

#### [NEW] `backend/models/responses.py`
Defines `ApiResponse[T]` to strictly enforce `{ data, error, meta }` across all endpoints.

---
### 3. Auth Standardization & Hardening

#### [MODIFY] `backend/auth/deps.py`
Update `AuthUser` model to strictly match:
```python
class AuthUser:
    user_id: str
    username: str
    role: str

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"
```

#### [MODIFY] `backend/models/schemas.py`
Update `AuthMeResponse` to align with `AuthUser`.

---
### 4. Database & Repository Fixes

#### [MODIFY] `backend/repositories/products.py`
Remove `InMemoryProductRepository`. Verify index on `owner_user_id` + `created_at`.

#### [MODIFY] `backend/repositories/users.py`
Remove `InMemoryUserRepository`.

---
### 5. Tools & Agents

#### [MODIFY] `backend/tools/crm_tool.py`
Eliminate mutable global dictionaries (`_updates`) by introducing thread-safe `threading.Lock` and enforcing deep copies correctly. Adding tool permission controls in the calling context.

---

## Open Questions

> [!IMPORTANT]
> **API Wrapping**: Requirement 5 asks for a wrapper `{ data: ..., error: null, meta: ... }`. Do we need to update `schemas.py` to wrap `ProductResponse` inside an `AppResponse(BaseModel)` or can I just return dictionaries/JSONResponses from the router? I will build a Generic `APIResponse` model to type-hint the returns correctly. Is this acceptable?
> **Error Types**: Are there specific HTTP codes you want mapped for `APIError` inside the global exception handler besides the existing mappings?

## Verification Plan

### Automated Tests
- `pytest backend/tests/` to verify that JWT authentication, product CRUD, and auth logic hold up after the refactor.
- Manually run application with `uvicorn backend.main:app` and test `/api/v1/auth/login` and `/api/v1/products`.

### Manual Verification
- Register a new user, log in, refresh token.
