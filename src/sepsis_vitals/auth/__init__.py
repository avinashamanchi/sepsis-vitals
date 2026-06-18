"""
sepsis_vitals.auth — Authentication and authorisation subsystem.

Public API
----------
- **jwt** — Password hashing, RBAC, TOTP, and account-lockout helpers.
- **tokens** — JWT access/refresh token creation and verification.
- **service** — High-level auth operations (register, login, reset, verify).
- **middleware** — FastAPI dependencies for auth, roles, and org access.
- **router** — FastAPI ``APIRouter`` with ``/auth`` endpoints.
"""

from __future__ import annotations
