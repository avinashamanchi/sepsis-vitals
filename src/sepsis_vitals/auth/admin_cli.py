"""Operator commands that need direct database access.

    python -m sepsis_vitals.auth.admin_cli mfa-status [--roles system_admin]
    python -m sepsis_vitals.auth.admin_cli reset-mfa --email someone@example.org
    python -m sepsis_vitals.auth.admin_cli create-user --email a@site.org --role nurse --site WARD-A

create-user provisions an account, including the first system_admin (the
public registration endpoint never creates admins). The password is read
from an interactive prompt, or from standard input with --password-stdin;
it is never accepted as an argument, printed or logged.

mfa-status is read-only: per role, how many accounts exist, how many
have MFA on, and how many hold recovery codes. With --roles (or the
configured SEPSIS_MFA_REQUIRED_ROLES) it also counts the accounts that would
be limited to MFA enrollment if enforcement were switched on. Counts only;
no account is named.

For the last-administrator case: resets MFA for one account and ends its
sessions. Requires the deployment's DATABASE_URL and SEPSIS_PII_KEY (the
email lookup uses the blind index). The action is written to the audit log;
no secrets are printed.
"""

from __future__ import annotations

import argparse
import logging
import sys


def mfa_status(db, roles: frozenset) -> dict:
    """Counts per role, and the accounts enforcement for *roles* would hold back."""
    import json

    from sepsis_vitals.db import User

    status: dict = {"roles": {}, "enforced_roles": sorted(roles), "would_need_enrollment": 0}
    for user in db.query(User):
        entry = status["roles"].setdefault(user.role, {"accounts": 0, "mfa_enabled": 0, "with_recovery_codes": 0})
        entry["accounts"] += 1
        entry["mfa_enabled"] += int(bool(user.mfa_enabled))
        entry["with_recovery_codes"] += int(bool(json.loads(user.mfa_recovery_hashes or "[]")))
        if user.role in roles and not user.mfa_enabled:
            status["would_need_enrollment"] += 1
    status["ready_to_enforce"] = bool(roles) and status["would_need_enrollment"] == 0
    return status


OPERATOR_ROLES = ("nurse", "researcher", "system_admin")


def create_user(db, email: str, password: str, role: str, site_id: str | None):
    """Create one account (strength policy, duplicate check, blind index)."""
    from sepsis_vitals.auth.service import (
        DuplicateEmailError,
        _validate_password_strength,
        compute_blind_index,
        hash_password,
    )
    from sepsis_vitals.db import User
    from sepsis_vitals.security import blind_index_candidates

    if role not in OPERATOR_ROLES:
        raise ValueError(f"role must be one of {', '.join(OPERATOR_ROLES)}")
    if role != "system_admin" and not site_id:
        raise ValueError("--site is required for site-scoped roles")
    _validate_password_strength(password)
    if db.query(User).filter(User.email_hash.in_(blind_index_candidates(email))).first() is not None:
        raise DuplicateEmailError("an account with that email already exists")
    user = User(email=email, email_hash=compute_blind_index(email), password_hash=hash_password(password),
                role=role, site_id=site_id)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _read_password(from_stdin: bool) -> str:
    import getpass

    if from_stdin:
        return sys.stdin.readline().rstrip("\n")
    first = getpass.getpass("Password: ")
    if getpass.getpass("Repeat password: ") != first:
        raise ValueError("passwords do not match")
    return first


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    reset = sub.add_parser("reset-mfa", help="clear MFA for one account")
    reset.add_argument("--email", required=True)
    status_cmd = sub.add_parser("mfa-status", help="read-only MFA enrollment counts per role")
    status_cmd.add_argument("--roles", help="comma-separated roles to evaluate (default: configured policy)")
    create = sub.add_parser("create-user", help="provision an account (password prompted, never an argument)")
    create.add_argument("--email", required=True)
    create.add_argument("--role", required=True, choices=OPERATOR_ROLES)
    create.add_argument("--site", default=None, help="site id (required for nurse and researcher)")
    create.add_argument("--password-stdin", action="store_true", help="read the password from standard input")
    opts = parser.parse_args(argv)

    if opts.cmd == "create-user":
        from sepsis_vitals.auth.service import AuthServiceError, DuplicateEmailError
        from sepsis_vitals.db import SessionLocal, init_db

        try:
            password = _read_password(opts.password_stdin)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        init_db()
        db = SessionLocal()
        try:
            user = create_user(db, opts.email, password, opts.role, opts.site)
        except (ValueError, DuplicateEmailError, AuthServiceError) as exc:
            print(f"Not created: {exc}", file=sys.stderr)
            return 1
        finally:
            db.close()
        logging.getLogger("sepsis_vitals.auth.admin_cli").warning(
            "AUDIT user_created operator_cli user=%s role=%s", user.id, user.role
        )
        print(f"Created {opts.role} account {user.id}.")
        return 0

    if opts.cmd == "mfa-status":
        import json

        from sepsis_vitals.auth.service import mfa_required_roles
        from sepsis_vitals.db import SessionLocal

        roles = frozenset(r.strip() for r in opts.roles.split(",") if r.strip()) if opts.roles \
            else mfa_required_roles()
        db = SessionLocal()
        try:
            print(json.dumps(mfa_status(db, roles), indent=2))
        finally:
            db.close()
        return 0

    from sepsis_vitals.auth.mfa import reset_mfa
    from sepsis_vitals.db import SessionLocal, User
    from sepsis_vitals.security import blind_index_candidates

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email_hash.in_(blind_index_candidates(opts.email))).first()
        if user is None:
            print("No account with that email.", file=sys.stderr)
            return 1
        reset_mfa(user, db)
        logging.getLogger("sepsis_vitals.auth.admin_cli").warning(
            "AUDIT mfa_reset operator_cli user=%s", user.id
        )
        print(f"MFA reset for user {user.id}; all sessions revoked.")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
