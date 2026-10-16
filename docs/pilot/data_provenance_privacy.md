# Data provenance and privacy assessment

**Status:** engineering assessment for review by the privacy officer. It is
not a completed DPIA or a HIPAA risk analysis.

## Data used so far

| Data | Source | Contains real patients? | Where |
|---|---|---|---|
| Model training and every report | Synthetic generator (`ml/synthetic_data.py`), fixed seeds | No | Regenerable from code; dataset digests in report provenance |
| Test fixtures | Synthetic, hand-written | No | `tests/` |
| MIMIC-IV demo files | PhysioNet open-access demo (de-identified real ICU data) | **Yes (de-identified)** | Only on a developer workstation (`physionet.org/`, git-ignored, never committed). Some loader tests read it if present; CI does not have it. Not used for any model or report. |
| Full MIMIC-IV | — | — | Not accessed; no credentials or agreement (M5) |

Before real or de-identified patient data is used for anything in this
project, **explicit approval from the project owner** is required, together
with a data-use agreement and ethics approval. The MIMIC-IV demo files on
the workstation are open-access, but they fall under the same rule for any
new use.

## Data the running system processes

| Data | Stored where | Protection |
|---|---|---|
| Patient identifiers (MRN, name, DOB) | `patients` | Field encryption (AES-GCM, versioned keyring, `pii_keys.py`); blind index for lookup |
| Vital signs and labs | `vitals` | Database encryption at rest (RDS + KMS in Terraform) |
| Prediction inputs and outputs | `prediction_records` | At rest as above. Append-only, sealed. `patient_id` is free text, stored as given: **callers must send an internal pseudonymous id, not an MRN.** This is a pilot requirement and is not enforced in code. |
| Access audit | `audit_log` and structured log `HIPAA_AUDIT` | Patient UUID only. MRNs in paths are replaced by a keyed reference. |
| Request log | stdout (`sepsis_vitals.requests`) | Route template, status, latency, model version, outcome and error class. **No identifiers or bodies** (tested). |
| Secrets | AWS Secrets Manager (KMS) / environment | Never logged. The local `.claude/settings.local.json` has credentials that were never committed; see PROJECT_REVIEW.md. |

## PHI controls (verified by tests unless noted)

- No PHI in logs:
  - request log, `test_request_log_has_route_and_outcome_but_no_identifiers`;
  - MFA logs, `test_mfa_replay`;
  - FHIR ingest logs, `test_fhir_ingest`.
- No PHI in fixtures, screenshots or reports. Every fixture is synthetic, and identifiers in tests are invented (for example `P001`, `synthetic-…`, `MRN-AUDIT-…`).
- A failed inference returns a fixed message with no patient data. Validation errors (422) return the offending field and value to the caller who sent it, as FastAPI does, and are not logged.
- Batch errors no longer echo patient ids or exception text. Up to 28e1e0a they did: `logger.warning(... patient.patient_id, exc)`, and the response included `str(exc)`.
- The copilot's external LLM path is off unless two flags and a BAA are in place. Only numeric vitals would be sent.

## Gaps (to close before real data)

1. `prediction_records.patient_id` is not encrypted and not validated as pseudonymous.
2. There is no retention policy. Periods are to be decided by the privacy officer and clinical governance. Append-only records need a privileged, audited purge procedure once a period is approved.
3. Access-audit persistence is best-effort after the response. Failures are now logged and counted (`sepsis_audit_log_failures_total`), and the stdout copy remains.
4. The GitHub Pages demo is public. It contains only the synthetic interface simulator, and publishing is now manual only.
5. There is no DPIA, and no data-flow diagram reviewed by the privacy officer.
