"""
sepsis_vitals.pii_rotation — inventory, re-key and verify PII under the keyring.

    python -m sepsis_vitals.pii_rotation inventory [--escalation-store PATH] [--json]
    python -m sepsis_vitals.pii_rotation rotate [--apply] [--batch-size 500] [--max-batches N]
                                                [--escalation-store PATH] [--encrypt-plaintext]
                                                [--skip-unreadable]
    python -m sepsis_vitals.pii_rotation verify [--escalation-store PATH]

Uses DATABASE_URL and the PII keyring (SEPSIS_PII_KEY, SEPSIS_PII_KEY_ID,
SEPSIS_PII_PREVIOUS_KEYS). See docs/pii_key_rotation.md for the procedure.

* ``rotate`` is a dry run unless ``--apply`` is given. Each batch is one
  transaction (rows are locked with FOR UPDATE on PostgreSQL), and each row
  is updated only if it still holds the value that was read. Interrupting
  the command loses at most the current batch; running it again continues
  with the rows not yet under the current key.
* Re-keying re-encrypts each value under the current key and recomputes
  its blind index from the decrypted value in the same update.
* Output names tables, columns, key IDs and counts. Rows are identified
  only by keyed references (``ref:...``). Values and keys are never printed.
* MFA recovery-code hashes and prediction-history references cannot be
  recomputed (the codes and identifiers are not stored). They keep working
  while the previous key stays configured. Retiring that key invalidates
  the recovery codes, so users must regenerate them first.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from sepsis_vitals.pii_keys import LEGACY_ID, Keyring, keyring_from_env, parse_token
from sepsis_vitals.security import log_ref


@dataclass(frozen=True)
class EncryptedColumn:
    table: str
    column: str
    pk: str = "id"
    index: Optional[str] = None  # blind-index column computed from this value

    @property
    def name(self) -> str:
        return f"{self.table}.{self.column}"


#: Every EncryptedString column in the ORM (tests check this list stays complete).
DB_COLUMNS: Tuple[EncryptedColumn, ...] = (
    EncryptedColumn("users", "email", index="email_hash"),
    EncryptedColumn("users", "totp_secret"),
    EncryptedColumn("patients", "external_id", index="external_id_hash"),
)

#: Sealed columns of the alert-escalation SQLite store (alert_escalation.v1.db).
ESCALATION_COLUMNS: Tuple[EncryptedColumn, ...] = (
    EncryptedColumn("tracked_alerts", "patient_id", pk="alert_id"),
    EncryptedColumn("alert_audit_trail", "user_id"),
    EncryptedColumn("alert_audit_trail", "detail"),
)


class RotationError(RuntimeError):
    """Stops a run. Messages never contain values or key material."""


# -- storage backends ------------------------------------------------------------------

class _Store:
    """Minimal raw-SQL access (no ORM: the ORM would decrypt and re-encrypt)."""

    lock_rows = False

    def batch(self, col: EncryptedColumn, exclude_prefix: str, after: Any, limit: int, lock: bool) -> List[Tuple]:
        raise NotImplementedError

    def update(self, col: EncryptedColumn, pk: Any, old: str, new: str, index: Optional[str]) -> bool:
        raise NotImplementedError

    def scan(self, col: EncryptedColumn) -> List[Tuple]:
        raise NotImplementedError

    def transaction(self) -> Any:
        raise NotImplementedError


class _SQLAlchemyStore(_Store):
    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.lock_rows = engine.dialect.name == "postgresql"
        self._conn: Any = None

    def _select(self, col: EncryptedColumn) -> str:
        return f"SELECT {col.pk}, {col.column}" + (f", {col.index}" if col.index else "") + f" FROM {col.table}"

    def batch(self, col, exclude_prefix, after, limit, lock):
        from sqlalchemy import text

        where = [f"{col.column} IS NOT NULL", f"{col.column} <> ''",
                 f"SUBSTR({col.column}, 1, :plen) <> :prefix"]
        params: Dict[str, Any] = {"plen": len(exclude_prefix), "prefix": exclude_prefix, "n": limit}
        if after is not None:
            where.append(f"{col.pk} > :after")
            params["after"] = after
        sql = self._select(col) + " WHERE " + " AND ".join(where) + f" ORDER BY {col.pk} LIMIT :n"  # nosec B608 - fixed identifiers
        if lock and self.lock_rows:
            sql += " FOR UPDATE"
        return [tuple(r) for r in self._conn.execute(text(sql), params)]

    def update(self, col, pk, old, new, index):
        from sqlalchemy import text

        sets = f"{col.column} = :new" + (f", {col.index} = :idx" if col.index and index is not None else "")
        result = self._conn.execute(
            text(f"UPDATE {col.table} SET {sets} WHERE {col.pk} = :pk AND {col.column} = :old"),  # nosec B608
            {"new": new, "idx": index, "pk": pk, "old": old},
        )
        return result.rowcount == 1

    def scan(self, col):
        from sqlalchemy import text

        with self.engine.connect() as conn:
            return [tuple(r) for r in conn.execute(text(self._select(col) + f" ORDER BY {col.pk}"))]  # nosec B608

    class _Txn:
        def __init__(self, store: "_SQLAlchemyStore") -> None:
            self.store = store

        def __enter__(self) -> None:
            self._ctx = self.store.engine.begin()
            self.store._conn = self._ctx.__enter__()

        def __exit__(self, *exc: Any) -> bool:
            self.store._conn = None
            return bool(self._ctx.__exit__(*exc))

    def transaction(self):
        return self._Txn(self)


class _SQLiteStore(_SQLAlchemyStore):
    """The escalation store: a plain SQLite file outside the main database."""

    def __init__(self, path: Path) -> None:
        if not path.exists():
            raise RotationError(f"escalation store not found: {path.name}")
        import sqlalchemy

        super().__init__(sqlalchemy.create_engine(f"sqlite:///{path}"))
        self.lock_rows = False


# -- classification ------------------------------------------------------------------

def _classify(keyring: Keyring, value: Optional[str]) -> str:
    if value is None or value == "":
        return "empty"
    if not value.startswith("enc:"):
        return "plaintext"
    try:
        parse_token(value)
    except ValueError:
        return "unreadable:malformed"
    kid = keyring.key_id_of(value)
    if kid is not None:
        return f"key:{kid}"
    stated, _ = parse_token(value)
    if stated is not None and keyring.get(stated) is None:
        return f"unreadable:key {stated} not configured"
    return "unreadable:no key decrypts"


def _index_status(keyring: Keyring, plaintext: Optional[str], stored: Optional[str]) -> str:
    if plaintext is None:
        return "unverifiable (value unreadable)"
    for i, key in enumerate(keyring.keys):
        if stored == keyring.blind_index(plaintext, key):
            return "current" if i == 0 else f"previous:{key.kid}"
    return "matches no key"


def _plaintext(keyring: Keyring, value: str) -> Optional[str]:
    if not value.startswith("enc:"):
        return value
    try:
        return keyring.decrypt(value)
    except ValueError:
        return None


def inventory(keyring: Keyring, store: _Store, columns: Tuple[EncryptedColumn, ...]) -> Dict[str, Any]:
    """Counts per column: storage format/key and blind-index state. No values."""
    report: Dict[str, Any] = {"current_key": keyring.current.kid,
                              "configured_keys": [k.kid for k in keyring.keys], "columns": {}}
    for col in columns:
        rows = store.scan(col)
        formats: Counter = Counter()
        indexes: Counter = Counter()
        for row in rows:
            value = row[1]
            formats[_classify(keyring, value)] += 1
            if col.index and value:
                indexes[_index_status(keyring, _plaintext(keyring, value), row[2])] += 1
        entry: Dict[str, Any] = {"rows": len(rows), "values": dict(sorted(formats.items()))}
        if col.index:
            entry["blind_index"] = dict(sorted(indexes.items()))
        report["columns"][col.name] = entry
    return report


def recovery_code_users(engine: Any) -> int:
    from sqlalchemy import text

    with engine.connect() as conn:
        return int(conn.execute(text(
            "SELECT COUNT(*) FROM users WHERE mfa_recovery_hashes IS NOT NULL "
            "AND mfa_recovery_hashes NOT IN ('', '[]')")).scalar() or 0)


# -- rotation --------------------------------------------------------------------------

@dataclass
class ColumnResult:
    rekeyed: int = 0
    encrypted_plaintext: int = 0
    skipped_plaintext: int = 0
    unreadable: int = 0
    changed_concurrently: int = 0
    batches: int = 0


def rotate(
    keyring: Keyring,
    store: _Store,
    columns: Tuple[EncryptedColumn, ...],
    *,
    apply: bool,
    batch_size: int = 500,
    max_batches: Optional[int] = None,
    encrypt_plaintext: bool = False,
    skip_unreadable: bool = False,
    progress: Callable[[str], None] = print,
) -> Dict[str, ColumnResult]:
    if keyring.current.kid == LEGACY_ID:
        raise RotationError(
            "the current key is 'legacy': set SEPSIS_PII_KEY to the new key with a new "
            "SEPSIS_PII_KEY_ID, and keep the old key as 'legacy:<key>' in SEPSIS_PII_PREVIOUS_KEYS"
        )
    if batch_size < 1:
        raise RotationError("--batch-size must be at least 1")
    current_prefix = f"enc:v2:{keyring.current.kid}:"
    results: Dict[str, ColumnResult] = {}
    batches_left = max_batches
    for col in columns:
        res = results.setdefault(col.name, ColumnResult())
        after: Any = None
        while batches_left is None or batches_left > 0:
            with store.transaction():
                rows = store.batch(col, current_prefix, after, batch_size, lock=apply)
                if not rows:
                    break
                for row in rows:
                    pk, value = row[0], row[1]
                    after = pk
                    if not value.startswith("enc:"):
                        if not encrypt_plaintext:
                            res.skipped_plaintext += 1
                            continue
                        plaintext: Optional[str] = value
                    else:
                        plaintext = _plaintext(keyring, value)
                    if plaintext is None:
                        res.unreadable += 1
                        if not skip_unreadable:
                            raise RotationError(
                                f"{col.name}: row {log_ref(pk)} cannot be decrypted with any configured key; "
                                "this batch was rolled back. Fix the keyring or pass --skip-unreadable."
                            )
                        progress(f"{col.name}: skipped unreadable row {log_ref(pk)}")
                        continue
                    if apply:
                        index = keyring.blind_index(plaintext) if col.index else None
                        if not store.update(col, pk, value, keyring.encrypt(plaintext), index):
                            res.changed_concurrently += 1
                            continue
                    if value.startswith("enc:"):
                        res.rekeyed += 1
                    else:
                        res.encrypted_plaintext += 1
                res.batches += 1
            if batches_left is not None:
                batches_left -= 1
            progress(f"{col.name}: batch {res.batches} done, {res.rekeyed + res.encrypted_plaintext} "
                     f"{'re-keyed' if apply else 'would be re-keyed'} so far")
    return results


def verify(keyring: Keyring, report: Dict[str, Any]) -> List[str]:
    """Problems that block retiring previous keys (empty list: all current)."""
    problems = []
    current = f"key:{keyring.current.kid}"
    for name, entry in report["columns"].items():
        for fmt, n in entry["values"].items():
            if fmt not in ("empty", current):
                problems.append(f"{name}: {n} value(s) {fmt}")
        for state, n in entry.get("blind_index", {}).items():
            # an unreadable value is already reported above
            if state not in ("current", "unverifiable (value unreadable)"):
                problems.append(f"{name}: {n} blind index(es) {state}")
    return problems


# -- CLI ------------------------------------------------------------------------------

def _stores(opts: argparse.Namespace) -> List[Tuple[_Store, Tuple[EncryptedColumn, ...]]]:
    from sepsis_vitals.db import engine

    stores: List[Tuple[_Store, Tuple[EncryptedColumn, ...]]] = [(_SQLAlchemyStore(engine), DB_COLUMNS)]
    if opts.escalation_store:
        stores.append((_SQLiteStore(Path(opts.escalation_store)), ESCALATION_COLUMNS))
    return stores


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sepsis_vitals.pii_rotation",
                                     description="Inventory, re-key and verify PII under the PII keyring.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("inventory", "rotate", "verify"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--escalation-store", help="path to alert_escalation.v1.db to include")
        cmd.add_argument("--json", action="store_true")
        if name == "rotate":
            cmd.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
            cmd.add_argument("--batch-size", type=int, default=500)
            cmd.add_argument("--max-batches", type=int, default=None, help="stop after N batches (resume later)")
            cmd.add_argument("--encrypt-plaintext", action="store_true",
                             help="also encrypt values stored unencrypted (from deployments without a key)")
            cmd.add_argument("--skip-unreadable", action="store_true",
                             help="report rows no configured key decrypts instead of stopping")
    opts = parser.parse_args(argv)

    keyring = keyring_from_env()
    if keyring is None:
        print("SEPSIS_PII_KEY is not set: nothing is encrypted with a key.", file=sys.stderr)
        return 2
    try:
        stores = _stores(opts)
        if opts.command == "rotate":
            for store, columns in stores:
                results = rotate(
                    keyring, store, columns, apply=opts.apply, batch_size=opts.batch_size,
                    max_batches=opts.max_batches, encrypt_plaintext=opts.encrypt_plaintext,
                    skip_unreadable=opts.skip_unreadable,
                    progress=(lambda m: None) if opts.json else print,
                )
                summary = {name: vars(r) for name, r in results.items()}
                print(json.dumps(summary, indent=2) if opts.json else
                      "\n".join(f"{n}: {r}" for n, r in summary.items()))
            if not opts.apply:
                print("Dry run: nothing was written. Re-run with --apply.")
            return 0
        reports = []
        for store, columns in stores:
            reports.append(inventory(keyring, store, columns))
        if stores and opts.command in ("inventory", "verify"):
            from sepsis_vitals.db import engine

            reports[0]["users_with_recovery_codes"] = recovery_code_users(engine)
        problems = [issue for rep in reports for issue in verify(keyring, rep)] if opts.command == "verify" else []
        if opts.json:
            print(json.dumps({"reports": reports, "problems": problems}, indent=2))
        else:
            for rep in reports:
                print(json.dumps(rep, indent=2))
            for issue in problems:
                print(f"NOT CURRENT: {issue}")
        if opts.command == "verify":
            print("verify: all values and blind indexes are under the current key" if not problems
                  else f"verify: {len(problems)} problem(s); previous keys are still needed")
            return 1 if problems else 0
        return 0
    except RotationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
