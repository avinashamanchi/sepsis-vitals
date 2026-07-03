# Sepsis-Vitals Code Audit — Claude Code Instructions

## Constraints (READ FIRST — NON-NEGOTIABLE)

1. **READ-ONLY AUDIT.** Do NOT edit, write, delete, or create any file. Do NOT run `git commit`, `git push`, `npm install`, `pip install`, or any command that modifies the workspace, dependencies, or git state.
2. **No tool installation.** Do NOT install CodeQL, Semgrep, SonarQube, Bandit, or any other tool. Use only the tools already available to you (Read, Grep, Glob, Bash for read-only commands like `git log`, `git blame`, `python -c`).
3. **Output only.** Your deliverable is a single audit report written to stdout. Organize findings by pass, with severity tags. Do not create files.
4. **No speculative fixes.** Report what you find and where (file:line). Do not propose diffs, refactors, or rewrites. The humans will decide what to fix.
5. **Time-budget awareness.** You have limited compute. Skip checks that return clean results quickly — spend time on findings, not on confirming things that are already correct.

---

## Project Context

- **Project:** sepsis-vitals (NOT "NextToken" — ignore that name if you see it)
- **Backend:** FastAPI + SQLAlchemy ORM + SQLite/PostgreSQL, Python 3.9+
- **Frontend:** React 19 + TypeScript 6 + Zustand + Vite, deployed to GitHub Pages
- **Auth:** JWT RS256 (key-pair, not shared secret) + Firebase (frontend)
- **Encryption:** AES-256-GCM at-rest for PII columns, PBKDF2/bcrypt for passwords
- **Entry points:** `src/sepsis_vitals/api.py` (backend), `frontend/src/main.tsx` (frontend)
- **Tests:** `tests/` directory, pytest

---

## PASS 0 — Structural Inventory (5 min)

**0.1** Generate a module dependency map. For each Python module under `src/sepsis_vitals/`, list what it imports and what imports it. Flag any module imported by >10 consumers or importing from >5 siblings.

**0.2** In `frontend/src/`, map component imports. Flag any component file >500 lines (monolith risk) or any circular import chains.

**0.3** Run `git log --oneline -50` and `git shortlog -sn` to estimate AI-generation ratio and iteration depth. Note but do not act on findings.

---

## PASS 1 — Architectural Integrity (10 min)

**1.1 — Dead module detection.** For every Python module under `src/sepsis_vitals/`, grep for import references across the codebase. Flag modules with zero non-test callers.

**1.2 — Orphan state (frontend).** In React components using `useState` or Zustand selectors: check that every `useEffect` with subscriptions, timers, or event listeners has a cleanup return. Check for Zustand selectors that call `.filter()`, `.map()`, or `.reduce()` inline (creates new references → infinite re-renders). *Note: one such bug was already fixed in `PatientDetail.tsx:AlertHistory` — check for remaining instances.*

**1.3 — Pattern consistency.** The backend uses a layered pattern: router → service → ORM. Verify all routers delegate to service functions rather than containing inline business logic. Flag any router that directly queries the database or performs multi-step mutations.

**1.4 — Abstraction audit.** Flag any abstract class or interface with exactly one implementation that adds no behavior. Focus on `src/sepsis_vitals/` Python code — the frontend uses a flat component model which is appropriate.

**1.5 — Dead code paths.** Grep for unreachable branches: `if False`, `if True`, variables assigned but never read, imports never used. In frontend: check for components exported but never imported anywhere.

---

## PASS 2 — Async Logic & State Machine Audit (10 min)

**2.1 — Unhandled async (backend).** Search for `async def` functions. For each, verify exceptions propagate to FastAPI's error handler rather than being silently caught. Flag any `except` block that logs and returns `None`/`undefined` without re-raising or returning a typed error response.

**2.2 — Unhandled async (frontend).** Search for `.then()` and `await` calls. Flag any `.catch(() => {})` (swallowed errors on non-trivial operations). *Note: some `.catch(() => {})` on optional API calls (forecast, simulator check) are acceptable — flag only those on critical paths like auth, vitals submission, or bundle operations.*

**2.3 — Race conditions.** Check:
- `src/sepsis_vitals/ml/state_store.py` — SQLite WAL mode, verify no concurrent write paths exist without transactions
- `src/sepsis_vitals/alerts/escalation.py` — thread-safe singleton, verify locking
- `frontend/src/lib/outbox.ts` — IndexedDB queue, verify flush() cannot overlap
- `src/sepsis_vitals/security.py` — in-memory rate limiter buckets, verify cleanup doesn't race with reads

**2.4 — Cleanup validation (frontend).** In `useWebSocket.ts`: verify the cleanup function closes the socket and clears reconnect timers. In `App.tsx`: verify activity listeners are removed on unmount. Check all `setInterval`/`setTimeout` calls have corresponding `clearInterval`/`clearTimeout`.

**2.5 — Empty/null collection handling.** For functions that process arrays of vitals, patients, or alerts: trace what happens when the input is empty or null. Focus on `ml/predictor.py`, `ml/forecast.py`, `scores.py`.

---

## PASS 3 — Security Audit (15 min — highest priority)

**3.1 — Secrets scan.** Grep the entire repo for patterns: API keys, passwords, tokens, or secrets assigned as string literals (not `os.environ`, `os.getenv`, `process.env`, or `import.meta.env`). Check `.env.example` for real values (should contain only `REPLACE_ME` placeholders). Check `frontend/` for any hardcoded Firebase config values beyond what's in `firebase.ts`.

**3.2 — Injection surfaces.** This project uses SQLAlchemy ORM (parameterized by default), so raw SQL injection is unlikely. Instead focus on:
- Any use of `text()`, `exec()`, `eval()`, `subprocess`, `os.system()`, or f-string SQL in Python
- Any use of `dangerouslySetInnerHTML` in React
- Any user input reaching file path construction (`os.path.join` with user-supplied segments)
- Template injection in Jinja2 or string formatting with user input
- The prompt injection guard in `security.py` — verify coverage of the `copilot` endpoint input

**3.3 — Auth & authorization completeness.** Map every FastAPI route (including those in routers). For each: (a) is auth middleware applied? (b) is resource-level authorization checked (does the user own this patient/alert/bundle)? Flag any endpoint that authenticates but doesn't authorize at the resource level (IDOR risk). Check the WebSocket `/ws/alerts` endpoint for auth enforcement.

**3.4 — CORS & headers.** Read the CORS config in `api.py`. Verify no wildcard `*` origin on authenticated endpoints. Verify security headers middleware is applied globally (not just to specific routes). *Known: this project already sets HSTS, CSP, X-Frame-Options, nosniff — verify they're still present and not weakened.*

**3.5 — Cryptographic audit.**
- Verify password hashing uses PBKDF2 (>=100K iterations) or bcrypt — check `auth/service.py`
- Verify JWT uses RS256 with key files, not HS256 with a string secret — check `auth/tokens.py`
- Verify AES-256-GCM encryption in `security.py:FieldEncryptor` uses proper nonce handling (unique per encryption)
- Check for any use of `MD5`, `SHA-1` for security purposes, or `random` (not `secrets`) for token generation

**3.6 — Dependency audit.** Read `pyproject.toml` and `frontend/package.json`. Flag any dependency that looks unfamiliar or has an unusual name (hallucination risk). Do NOT run `npm audit` or `pip audit` — just review the dependency lists for obvious issues.

---

## PASS 4 — Logic & Business Rule Integrity (10 min)

**4.1 — Clinical scoring correctness.** Read `src/sepsis_vitals/scores.py`. Verify qSOFA, SIRS, NEWS2 calculations against published criteria. Flag any threshold that doesn't match clinical literature.

**4.2 — Bundle protocol correctness.** Read `src/sepsis_vitals/bundles/protocol.py`. Verify the Hour-1 Bundle tasks match the Surviving Sepsis Campaign guidelines. Check conditional task logic (fluids gated on MAP/SBP, repeat lactate gated on initial lactate >2).

**4.3 — Return type consistency.** In service functions (`bundles/service.py`, `auth/service.py`, `patients/`): verify all code paths return the declared type. Flag functions that return `None` on error paths where the caller doesn't check for it.

**4.4 — Transaction atomicity.** In `bundles/service.py`: verify `start_bundle()` and `complete_task()` use proper transaction boundaries (`db.commit()` only after all mutations, `db.rollback()` on failure). Check that `expire_stale_bundles()` doesn't partially commit.

**4.5 — Risk level thresholds.** Verify consistency between backend risk classification (`ml/predictor.py`) and frontend risk display (`lib/risk.ts`, `PatientDetail.tsx:riskFromProb`). Flag any threshold mismatch.

---

## PASS 5 — Code Quality (5 min)

**5.1 — Duplication.** Spot-check for duplicate logic blocks >15 lines between: `api.py` route handlers and router files; scoring logic in `scores.py` vs `ml/predictor.py`; demo data generation across frontend pages.

**5.2 — High-complexity functions.** Identify functions >80 lines or with deeply nested conditionals (>4 levels). Focus on `api.py`, `ml/predictor.py`, `ml/trainer.py`.

**5.3 — Test quality.** Read 3-5 test files. Check whether tests assert specific behavioral outcomes or just assert "no exception thrown." Flag test files that mock the database when they should use the SQLite test fixture.

**5.4 — Logging audit.** Grep for `console.log`, `print(`, `logger.debug` that might output PII, tokens, passwords, or patient data. Focus on error handlers and auth flows.

---

## PASS 6 — AI-Specific Regression Patterns (5 min)

**6.1 — Naming consistency.** Spot-check variable naming conventions across files. Flag files that mix `camelCase` and `snake_case` in Python, or inconsistent component naming in TypeScript.

**6.2 — Context boundary seams.** Identify integration points between modules that use different patterns (different error handling styles, different abstraction levels). Focus on: bundles ↔ api.py, alerts/escalation ↔ alerts/router, ml/forecast ↔ api.py.

**6.3 — Phantom guards.** Flag `if` checks for conditions that cannot occur given the type system or ORM constraints (e.g., checking for `None` on a `NOT NULL` column, guarding against negative IDs on UUID fields).

---

## Output Format

Structure your report as:

```
# Sepsis-Vitals Audit Report

## Summary
- Critical: N findings
- High: N findings
- Medium: N findings
- Low: N findings
- Info: N findings

## PASS 0 — Structural Inventory
[findings or "Clean"]

## PASS 1 — Architectural Integrity
### [SEVERITY] Finding title
- **Location:** file:line
- **Issue:** description
- **Evidence:** code snippet or grep result

... (repeat for each pass)
```

## Severity Definitions (report-only — do NOT take action)

| Severity | Criteria |
|---|---|
| **Critical** | Exploitable auth bypass, secret exposure, injection vector, RCE surface |
| **High** | Swallowed error on critical path, missing authorization check, weak crypto |
| **Medium** | Orphan state, missing input validation on non-critical path, dead code |
| **Low** | Naming inconsistency, duplicate logic, excessive comments |
| **Info** | Cosmetic abstraction, phantom guard, style drift |
