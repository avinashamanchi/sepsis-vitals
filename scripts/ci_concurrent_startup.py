#!/usr/bin/env python3
"""Stress-test schema creation by several workers starting at the same instant.

Reproduces the multi-worker startup race (uvicorn ``--workers 4`` with every
worker calling ``init_db``): each round starts N processes on an empty
database, holds them at a barrier, then releases them into ``init_db``
together. Every worker must succeed and the schema must match the ORM.

    python scripts/ci_concurrent_startup.py --database-url sqlite:////tmp/race.db
    python scripts/ci_concurrent_startup.py --database-url postgresql://.../sepsis_startup_race

PostgreSQL rounds start from an empty ``public`` schema (DROP SCHEMA ...
CASCADE), so the script refuses any database whose name does not end in
``_startup_race``. SQLite rounds use a fresh file next to the given path.

``--attempts 1`` disables the retry, which shows whether the race actually
occurs on this machine (a meaningful test needs it to). Output contains
exception types and messages from DDL only, never row data.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import List, Tuple

SAFE_PG_SUFFIX = "_startup_race"


def _worker(url: str, barrier, attempts: int, results) -> None:
    os.environ["DATABASE_URL"] = url
    try:
        from sepsis_vitals import db

        db.engine.dispose()
        barrier.wait(timeout=120)
        db.init_db(production=False, attempts=attempts)
        results.put(("ok", ""))
    except Exception as exc:  # reported, not raised: the parent counts failures
        first_line = str(exc).strip().splitlines()[0][:200] if str(exc).strip() else ""
        results.put(("error", f"{type(exc).__name__}: {first_line}"))


def _reset(url: str, round_no: int) -> str:
    """An empty database for this round; returns the URL to use."""
    if url.startswith("sqlite:///"):
        base = Path(url[len("sqlite:///"):])
        path = base.with_name(f"{base.stem}-round{round_no}{base.suffix or '.db'}")
        if path.exists():
            raise SystemExit(f"refusing to reuse existing file {path}")
        return f"sqlite:///{path}"

    import sqlalchemy as sa

    from sepsis_vitals.db import sync_database_url

    sync_url = sync_database_url(url)
    name = sa.engine.make_url(sync_url).database or ""
    if not name.endswith(SAFE_PG_SUFFIX):
        raise SystemExit(f"refusing to drop the schema of '{name}': use a database named *{SAFE_PG_SUFFIX}")
    engine = sa.create_engine(sync_url)
    with engine.begin() as conn:
        conn.execute(sa.text("DROP SCHEMA public CASCADE"))
        conn.execute(sa.text("CREATE SCHEMA public"))
    engine.dispose()
    return url


def _verify(url: str) -> List[str]:
    os.environ["DATABASE_URL"] = url
    import sqlalchemy as sa

    from sepsis_vitals.db import Base, schema_drift, sync_database_url

    engine = sa.create_engine(sync_database_url(url))
    try:
        inspector = sa.inspect(engine)
        problems = [f"missing table {t.name}" for t in Base.metadata.sorted_tables if not inspector.has_table(t.name)]
        return problems + [f"missing column {c}" for c in schema_drift(engine)]
    finally:
        engine.dispose()


def run(url: str, workers: int, rounds: int, attempts: int) -> Tuple[int, List[str]]:
    """Returns (failed rounds, report lines)."""
    ctx = mp.get_context("spawn")
    failed, report = 0, []
    for round_no in range(1, rounds + 1):
        round_url = _reset(url, round_no)
        barrier = ctx.Barrier(workers)
        results = ctx.Queue()
        procs = [ctx.Process(target=_worker, args=(round_url, barrier, attempts, results)) for _ in range(workers)]
        for proc in procs:
            proc.start()
        outcomes = [results.get(timeout=300) for _ in procs]
        for proc in procs:
            proc.join(timeout=60)
        errors = [detail for status, detail in outcomes if status != "ok"]
        problems = _verify(round_url)
        ok = not errors and not problems
        failed += 0 if ok else 1
        report.append(
            f"round {round_no}: {workers - len(errors)}/{workers} workers ok"
            + ("" if not errors else f"; errors: {sorted(set(errors))}")
            + ("" if not problems else f"; schema: {problems[:5]}")
        )
    return failed, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--attempts", type=int, default=5, help="init_db retry budget (1 = no retry)")
    opts = parser.parse_args()

    failed, report = run(opts.database_url, opts.workers, opts.rounds, opts.attempts)
    print("\n".join(report))
    print(f"{opts.rounds - failed}/{opts.rounds} rounds clean ({opts.workers} workers, attempts={opts.attempts})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
