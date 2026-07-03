# Sepsis-Vitals Code Audit v2 — Claude Code Instructions

## Constraints (READ FIRST — NON-NEGOTIABLE)

1. **READ-ONLY AUDIT.** Do NOT edit, write, delete, or create any file. Do NOT run `git commit`, `git push`, `npm install`, `pip install`, or any command that modifies the workspace, dependencies, or git state.
2. **No tool installation.** Do NOT install CodeQL, Semgrep, SonarQube, Bandit, or any other tool. Use only the tools already available to you (Read, Grep, Glob, Bash for read-only commands like `git log`, `git blame`, `python -c`).
3. **Output only.** Your deliverable is a single audit report written to stdout. Organize findings by pass, with severity tags. Do not create files.
4. **No speculative fixes.** Report what you find and where (file:line). Do not propose diffs, refactors, or rewrites. The humans will decide what to fix.
5. **Time-budget awareness.** You have limited compute. Skip checks that return clean results quickly — spend time on findings, not on confirming things that are already correct.

---

## Project Context

- **Project:** sepsis-vitals
- **Repo:** https://github.com/avinashamanchi/sepsis-vitals.git
- **Backend:** FastAPI + SQLAlchemy ORM + SQLite/PostgreSQL, Python 3.9+
- **Frontend:** React 19 + TypeScript 6 + Zustand + Vite, deployed to GitHub Pages
- **Auth:** Firebase (Google + email/password) on frontend; JWT RS256 with HS256 fallback on backend (`auth/tokens.py`)
- **Encryption:** AES-256-GCM at-rest for PII columns, PBKDF2/bcrypt for passwords
- **Entry points:** `src/sepsis_vitals/api.py` (backend), `frontend/src/main.tsx` (frontend)
- **Tests:** `tests/` directory, pytest

## Prior Audit — Already Fixed (skip these)

A v1 audit was completed and the following findings were remediated. Do NOT re-flag these unless the fix is incomplete or introduced a new issue:

1. ~~IDOR / missing resource-level authz~~ → `verify_patient_org()` added to all patient/bundle/alert endpoints; WebSocket broadcasts scoped by org_id (`api.py`, `bundles/router.py`, `realtime/websocket.py`)
2. ~~JWT HS256 vs documented RS256~~ → `tokens.py` now loads RSA keys from env with HS256 fallback
3. ~~AUTH_ENABLED=false → silent admin in prod~~ → gated on `SEPSIS_ENV=production` (`api.py`)
4. ~~state_store.py shared SQLite connection without lock~~ → `threading.Lock` added (`ml/state_store.py`)
5. ~~Frontend re-derives risk instead of using backend risk_level~~ → uses `latest.risk_level ?? riskFromProb()` (`PatientDetail.tsx`)
6. ~~React error #185 infinite re-render~~ → Zustand `.filter()` in selector fixed (`PatientDetail.tsx:AlertHistory`)
7. ~~Duplicate JWT implementation~~ → `jwt.py` token functions are dead code, only hash/lockout functions used (known, not yet removed)

## What to Focus On This Time

This is a **second-pass deep dive**. Go deeper than the v1 audit. Focus on:

- **Things the v1 audit flagged as Medium/Low/Info that weren't fixed** (e.g., NEWS2 Scale 2 gap, dead modules, `except (TokenError, Exception)` redundancy)
- **New code introduced by the security fixes** — verify the fixes themselves don't introduce bugs (e.g., does `verify_patient_org()` handle edge cases correctly? Does the RS256/HS256 key loading have race conditions?)
- **Business logic correctness** — go deeper on clinical scoring, bundle protocol, and ML pipeline logic
- **Frontend robustness** — look beyond PatientDetail; audit all pages for async issues, missing error handling, stale state
- **Data flow integrity** — trace user input from API entry to database write to frontend display
- **Test coverage gaps** — identify critical paths that have no test coverage

---

## PASS 0 — Verify Prior Fixes (5 min)

**0.1** Spot-check each of the 6 remediated findings listed above. For each: read the relevant file and confirm the fix is correctly implemented. Flag if a fix is incomplete, has edge cases, or introduced a new issue.

**0.2** Check `verify_patient_org()` in `api.py`: What happens when `patient_id` doesn't exist in the Patient table but exists in other tables (e.g., bundles, vitals)? Is there a foreign key inconsistency risk?

**0.3** Check `tokens.py` key loading: Is `_load_keys()` thread-safe? Could two requests race on `_KEYS_LOADED`?

---

## PASS 1 — Deep Architectural Audit (10 min)

**1.1 — Monolith analysis.** `api.py` is 1527+ lines. Inventory all responsibilities it contains (routes, middleware, Pydantic models, copilot logic, WebSocket handler). Identify which responsibilities should be in separate modules. Count the number of route handlers, middleware functions, and Pydantic models in this one file.

**1.2 — Cross-module contract validation.** For each router module (`bundles/router.py`, `alerts/router.py`, `patients/router.py`, `auth/router.py`): verify the Pydantic request/response models match what the frontend sends/expects. Check for field name mismatches, missing optional fields, or type disagreements.

**1.3 — Database model integrity.** Read `db.py` and all migration files. Verify: (a) every foreign key has an index, (b) cascade deletes are configured correctly, (c) no orphan records can be created by the current API routes.

**1.4 — Import cycle detection.** The IDOR fix added `from sepsis_vitals.db import Patient` inside function bodies (deferred imports). Trace all deferred imports and verify none create circular dependency issues.

**1.5 — Dead code inventory.** List ALL modules, functions, and classes that are never called from production code paths. Include: `jwt.py` token functions, `fhir/listener.py`, `state.py`, `model_scaffold.py`, `data_quality.py`, and anything else found. Quantify the dead code as a percentage of total codebase.

---

## PASS 2 — Async & Concurrency Deep Dive (10 min)

**2.1 — FastAPI async/sync mixing.** FastAPI runs `async def` handlers on the event loop and `def` handlers in a threadpool. Identify any `async def` handler that calls synchronous blocking code (e.g., `db.query()`, `time.sleep()`, file I/O) without using `run_in_executor`. This blocks the event loop.

**2.2 — Database session lifecycle.** Trace every `SessionLocal()` call. Verify every session is closed in a `finally` block or context manager. Look for sessions opened in route handlers that could leak on exception. Check the new `verify_patient_org()` sessions added in the IDOR fix.

**2.3 — WebSocket connection leak.** In `realtime/websocket.py`, the IDOR fix added `_patient_org_id()` which opens a DB session inside `broadcast()`. Since `broadcast()` is called for every message to every client, verify: (a) the session is always closed, (b) it doesn't create a performance bottleneck, (c) what happens if the DB query fails mid-broadcast.

**2.4 — Rate limiter atomicity.** In `security.py`, the in-memory `RateLimiter` uses a `dict` for buckets. Under FastAPI's threadpool, concurrent requests could corrupt the bucket dict. Verify thread safety. Compare with the Redis-backed path which uses Lua scripts for atomicity.

**2.5 — Background task safety.** Check `monitoring/drift_monitor.py` background task. Verify it handles exceptions without crashing the server, doesn't leak memory in its rolling buffer, and doesn't conflict with the main request-handling threads.

---

## PASS 3 — Security Deep Dive (15 min)

**3.1 — Firebase config exposure.** `frontend/src/lib/firebase.ts` now embeds Firebase config values as fallback defaults. Verify these are genuinely public-safe (API key, project ID, etc.) and that no server-side secrets are exposed. Check that Firebase Security Rules are the actual security boundary, not client-side checks.

**3.2 — Session management.** Trace the full auth flow: Firebase issues a token → frontend stores it → frontend sends it to backend. How does the backend validate Firebase tokens? Does it verify the token signature against Firebase's public keys, or does it just decode without verification? Check `auth/middleware.py` and `auth/tokens.py`.

**3.3 — Input validation completeness.** For each API endpoint that accepts user input: verify Pydantic models enforce type constraints, string length limits, and value ranges. Focus on: vitals input (can you submit heart_rate=99999?), patient creation, alert actions, bundle operations. Check for missing validation that could corrupt data or crash the ML model.

**3.4 — Rate limiting bypass.** Check if the rate limiter can be bypassed by: (a) using different IP addresses (X-Forwarded-For spoofing), (b) sending requests without auth (do unauthenticated requests hit the limiter?), (c) WebSocket connections (are they rate-limited?).

**3.5 — CORS configuration edge cases.** Read the CORS config in `api.py`. Check: what happens when `SEPSIS_ALLOWED_ORIGINS` is not set? Is there a default that's too permissive? Does the wildcard stripping logic handle edge cases like `*` embedded in a longer string?

**3.6 — Prompt injection depth.** The copilot endpoint has prompt injection detection. Read the regex patterns in `security.py`. Try to identify bypasses: unicode homoglyphs, zero-width characters, base64-encoded payloads, nested injection attempts. How robust is `_deidentify_vitals()`?

**3.7 — Encryption key management.** If `SEPSIS_PII_KEY` is not set, `FieldEncryptor` falls back to plaintext. Verify there's a startup warning or check. What happens if the key is rotated — can existing encrypted data still be read? Is there a key rotation mechanism?

---

## PASS 4 — Business Logic Deep Dive (10 min)

**4.1 — ML prediction pipeline.** Trace a prediction request from API entry (`/predict`) through feature engineering, model inference, and response. Verify: (a) input features are validated before reaching the model, (b) model output is bounded (probabilities between 0 and 1), (c) SHAP explanations can't leak training data.

**4.2 — Deterioration detection correctness.** Read `ml/monitor.py` and `ml/forecast.py`. Verify the deterioration detection thresholds, trend calculations, and alert generation logic. Are the EWMA parameters clinically reasonable? Could false positives flood alerts?

**4.3 — Bundle state machine integrity.** Read `bundles/service.py`. Trace all state transitions: open → completed, open → expired, open → cancelled. Verify: (a) no invalid transitions are possible, (b) concurrent task completions can't corrupt bundle state, (c) the auto-complete logic (`_all_critical_done`) handles race conditions.

**4.4 — Score calculation edge cases.** For each scoring function in `scores.py`: what happens with extreme vitals (HR=0, temp=45, SBP=-1, GCS=0)? Are there boundary conditions that produce NaN, infinity, or division by zero?

**4.5 — Simulator data isolation.** The simulator generates synthetic patients. Verify simulator data cannot leak into real patient data paths, real alerts, or real prediction records. Check `ml/simulator.py` integration points.

---

## PASS 5 — Frontend Deep Dive (10 min)

**5.1 — All pages async audit.** For every page component (Dashboard, Patients, Monitor, ScoreLab, Predict, Analytics, Alerts, Admin, Population): check that every `useEffect` with API calls handles loading, error, and empty states. Flag any `.catch(() => {})` on critical paths.

**5.2 — Store mutation safety.** Read `stores/useStore.ts`. Check for Zustand selectors that create new references (`.filter()`, `.map()`, object spread in selectors). We fixed one in AlertHistory — check all other consumers.

**5.3 — XSS surface.** Search for any `dangerouslySetInnerHTML` in project source (not vendored/build output). Search for any user input rendered without escaping. Check if alert messages, patient notes, or copilot responses could contain HTML/script.

**5.4 — Auth state consistency.** Trace what happens when: (a) Firebase token expires mid-session, (b) user signs out in another tab, (c) network goes offline then online. Does the app handle these gracefully or show stale/broken state?

**5.5 — Accessibility basics.** Spot-check 3 pages for: missing aria labels on interactive elements, form inputs without labels, color-only risk indicators (no text/icon fallback for colorblind users), keyboard navigation traps.

---

## PASS 6 — Test Coverage Analysis (5 min)

**6.1 — Critical untested paths.** Identify the 5 most critical code paths that have NO test coverage. Prioritize: auth flows, patient data CRUD, prediction pipeline, bundle state machine, alert escalation.

**6.2 — Test isolation.** Check if tests share global state (singleton instances, module-level variables) that could make test order matter. Check if the `PatientStateStore` singleton or `AlertEscalationManager` singleton could leak state between tests.

**6.3 — Frontend test existence.** Does any frontend test infrastructure exist (Jest, Vitest, Playwright, Cypress)? If not, flag as a gap.

---

## Output Format

```
# Sepsis-Vitals Audit Report v2

## Summary
- Critical: N findings
- High: N findings
- Medium: N findings
- Low: N findings
- Info: N findings

## Prior Fix Verification
[confirm each fix or flag issues]

## PASS 1 — Architectural Audit
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
| **High** | Swallowed error on critical path, missing authorization check, weak crypto, data corruption risk |
| **Medium** | Missing input validation, dead code with security implications, performance bottleneck, missing test coverage on critical path |
| **Low** | Naming inconsistency, duplicate logic, excessive comments, minor UX issue |
| **Info** | Cosmetic abstraction, phantom guard, style drift, documentation gap |
