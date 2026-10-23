# Security and privacy checklist

Status of each control, with how it is verified. "Repo" means it is
implemented in code or configuration and has not been applied to any
environment. Nothing here was applied to AWS or to the GitHub repository
settings.

## Access control

| Control | Status | Evidence |
|---|---|---|
| Authentication on every data route | Done | `verify_auth`; it cannot be disabled in production (`dependencies.py`); tests |
| Site (tenant) isolation, including every scoring path | Done | `tests/test_tenant_isolation.py` (PostgreSQL in CI); `test_foreign_patients_are_refused_before_scoring_or_recording` |
| Roles and least privilege in the API | Done | Admin-only alert tests; role checks (E2E) |
| MFA (TOTP) with replay protection and recovery codes | Done | N52; `tests/test_mfa_replay.py`; browser E2E |
| **MFA enforcement** | **Not enabled** (approval item) | `SEPSIS_MFA_REQUIRED_ROLES` is empty by default. The owner chooses the roles. Existing users must enrol first. |
| Session lifetime, refresh rotation, reuse detection, logout revocation | Done | E2E |
| Break-glass | Disabled (403) | Needs a policy (R10) |
| IAM least privilege (ECS) | Repo | Execution role: ECR pull, logs, named secrets and `kms:Decrypt` only. The task role has no policies. Not planned against an account. |

## Data protection

| Control | Status | Evidence |
|---|---|---|
| TLS in transit | Repo | ALB HTTPS listener; HSTS header (tests) |
| Encryption at rest | Repo | RDS `storage_encrypted` with KMS; Secrets Manager with KMS; S3 model bucket with KMS |
| Field encryption for identifiers | Done | `pii_keys.py`, versioned keyring, rotation CLI tested on PostgreSQL in CI. **Rotating production keys is an approval item.** |
| No PHI in logs or errors | Done (tested) | `docs/pilot/data_provenance_privacy.md` |
| Append-only, sealed prediction audit | Done | Migration 006 triggers; `verify_record`; PostgreSQL CI and compose E2E |
| Access audit (who saw which patient) | Done; best-effort persistence | `audit_log` plus a stdout copy; failures counted |
| Retention policy | **Missing** | See below |
| Backups | Repo | RDS backup retention of 30 days, deletion protection, Multi-AZ. Restore not tested. |

## Supply chain and repository

| Control | Status | Evidence |
|---|---|---|
| Hashed dependency locks, pip-audit, npm audit, Dependabot | Done | CI `security` job |
| Bandit, ruff, mypy, eslint, tsc | Done | CI |
| Secrets not in the repository | Done | Checked. Two credentials in the untracked local `.claude/settings.local.json` were never committed (PROJECT_REVIEW.md); **revoking them is the owner's action.** |
| Branch protection on `main` | **Draft only, not applied** | `docs/pilot/branch_protection_main.json` |
| Pages deployment | **Manual only** (this change) | `.github/workflows/pages.yml` runs only on `workflow_dispatch` |
| Private vulnerability reporting | Disabled (owner action) | A11b |

## Retention policy (draft structure; the periods are not decided)

| Record | Proposed rule | Period |
|---|---|---|
| `prediction_records` | Keep for the pilot plus the regulatory retention period; purge only through a privileged, audited procedure that keeps a tombstone (id, digest, purge authority) | To be decided by the privacy officer and clinical governance |
| `audit_log` | As for the hospital's access-audit retention | To be decided |
| `vitals`, `patients` | As for the data-use agreement | To be decided |
| Request logs (CloudWatch) | Operational only (no identifiers) | Terraform currently sets 90 days (engineering default); to be confirmed |
| Backups | Follow the source tables | To be decided |

## Incident process (draft)

1. **Detect.** CloudWatch 5xx and DB alarms (SNS), `/ready` failures, `sepsis_audit_log_failures_total`, `recent_inference` success rate in `/model/status`, or a user report.
2. **Contain.**
   - Scale the service to zero, or roll back (`docs/pilot/runbook.md`).
   - Revoke sessions and rotate keys **with owner approval**.
   - For a suspected PHI exposure, stop log export and preserve the evidence.
3. **Notify** the named incident owner, the privacy officer and the clinical lead (names to be filled in before the pilot). PHI incidents follow the hospital's breach procedure and its deadlines.
4. **Investigate** using the request id, `prediction_records` (sealed) and `audit_log`.
5. **Recover and review:** a post-incident review, and a risk-register update.

## Before a pilot

- Penetration test.
- DPIA.
- Threat model reviewed by the hospital's security team.
- MFA enforced for every clinical role.
- Branch protection applied.
- Retention periods set.
- Incident owners named.
- Restore drill completed.
