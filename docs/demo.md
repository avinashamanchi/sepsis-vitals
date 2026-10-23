# Demo plan (synthetic data only)

This demo shows the prototype working end to end: validation, scoring,
explanation, alert decision, audit record, and failure handling. **Use
synthetic patients only.** Never enter real patient data. The software is
not approved for clinical use.

Every step below was run on 2026-10-16 against a local server.

## 1. Start

```bash
python -m venv .venv && . .venv/bin/activate
pip install --require-hashes -r requirements/dev.txt && pip install --no-deps -e .
```

API (development mode, local only, authentication off, throwaway database):

```bash
SEPSIS_ENV=development SEPSIS_AUTH_ENABLED=false DATABASE_URL=sqlite:///./demo.db \
SEPSIS_STATE_DIR=./demo-state SEPSIS_MODEL_DIR=models \
python -m uvicorn sepsis_vitals.api:app --host 127.0.0.1 --port 8765 2>&1 | tee demo-server.log
```

Optionally, the full stack with UI: `docker compose -f docker/docker-compose.yml up`
(see `docker/` for the required local secrets). The UI's Predict page shows
the same fields as the API.

## 2. Run the cases

```bash
python scripts/demo_synthetic.py --base http://127.0.0.1:8765
```

| Case | What happens | Expected |
|---|---|---|
| Status | `/health`, `/ready`, `/model/status` | Ready, audit guard present, artifact verified, schema compatible, `clinically_ready: false` |
| High risk | A deteriorating 4-hour synthetic timeline (`synthetic-demo-high`, age 78) | Level critical, alert true, policy `placeholder-bands-v1`, per-patient attribution, data-quality warnings, `audit_record_id` |
| Low risk | A stable timeline (`synthetic-demo-low`, age 34) | Level low, no alert, recorded |
| Malformed | Unordered timestamps, a Fahrenheit declaration, an unknown field, a timestamp without an offset, an out-of-range value | Each refused with 422 and the reason; nothing recorded |

On 2026-10-16 every check passed. The high-risk example's attribution was
dominated by procalcitonin, which shows the synthetic model's behaviour
(N53). It is not a clinical insight.

## 3. Backend failure

Start a second server without a model, on port 8766 with
`SEPSIS_MODEL_DIR=<an empty directory>`, then:

```bash
python scripts/demo_synthetic.py --base http://127.0.0.1:8766 --case failure
```

Expected: 503, with "Predictions are unavailable: no usable model is
installed" and no score.

Audit-store failure: make the database directory read-only (`chmod a-w`),
send one request, then restore it (`chmod u+w`). Expected: 503 "No risk
calculated: the result could not be written to the audit trail, so it was
withheld. This is not a low-risk result."

## 4. Inspect provenance and history

```bash
DATABASE_URL=sqlite:///./demo.db python -m sepsis_vitals.audit_records verify
DATABASE_URL=sqlite:///./demo.db python -m sepsis_vitals.audit_records show <audit_record_id>
DATABASE_URL=sqlite:///./demo.db python -m sepsis_vitals.audit_records show <audit_record_id> --include-content
```

- `verify` reports how many records are intact, altered or unsealed.
- `show` gives source, request id, threshold policy, input and record digests, seal state, and model, artifact, feature-schema, preprocessing and policy versions.
- `--include-content` (synthetic demos only) adds the full input and output.

Other views:

- The `X-Request-ID` response header matches `request_id` in the record and in the request log.
- Patient trend: `GET /patient/<id>/trend`.

## 5. Verify no PHI

```bash
grep -c "synthetic-demo" demo-server.log
```

Expected: `0`. The request log carries route templates, statuses, latency,
outcomes and error classes only. On 2026-10-16 there were no matches.

## 6. Stop

Press Ctrl-C (the server shuts down cleanly). Then delete `demo.db` and
`demo-state/` if you no longer need them; they hold only synthetic records.
