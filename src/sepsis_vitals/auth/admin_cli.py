"""Operator commands that need direct database access.

    python -m sepsis_vitals.auth.admin_cli reset-mfa --email someone@example.org

For the last-administrator case: resets MFA for one account and ends its
sessions. Requires the deployment's DATABASE_URL and SEPSIS_PII_KEY (the
email lookup uses the blind index). The action is written to the audit log;
no secrets are printed.
"""

from __future__ import annotations

import argparse
import logging
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    reset = sub.add_parser("reset-mfa", help="clear MFA for one account")
    reset.add_argument("--email", required=True)
    opts = parser.parse_args(argv)

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
