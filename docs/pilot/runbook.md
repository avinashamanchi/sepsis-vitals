# Operations runbook

For the Terraform/ECS deployment (`terraform/`) and the Docker Compose stack
(`docker/`). **No environment has been deployed by this project.** Every
AWS step below needs credentials and the owner's approval.

## 1. Deploy (ECS)

Migrations must be backward compatible: the running version has to keep
working on the migrated schema (expand first, contract later). Migrations
005 and 006 are additive. 006 also forbids UPDATE and DELETE on
`prediction_records`, which no version ever did.

1. Build and push the image `sepsis-vitals:<tag>` from a commit whose CI is green.
2. Register the new task definition only:
   `terraform -chdir=terraform apply -target=aws_ecs_task_definition.api -var image_tag=<tag>`
3. **Pre-serve migration** with the new revision:
   `python scripts/ecs_migrate.py --from-terraform terraform --execute`
   (without `--execute` it is a dry run). Stop if it reports a failure.
4. Update the service: `terraform -chdir=terraform apply -var image_tag=<tag>`.
5. Watch the rollout:
   - The ALB health check is `/ready`. It needs the database, the migration head and the audit guard.
   - The ECS circuit breaker rolls back a deployment that never becomes healthy.
6. Smoke-check, using synthetic data only:
   - `/model/status`: `state: ready`, checks `verified` and `compatible`, `clinical_use: not-permitted`;
   - one synthetic `/predict/timeseries` request, whose `audit_record_id` should exist.

Compose: `docker/entrypoint.sh` runs `alembic upgrade head` before starting,
unless `SEPSIS_RUN_MIGRATIONS=false`.

## 2. Health checks

| Endpoint | Meaning | Healthy |
|---|---|---|
| `/health` | Process alive (liveness). Never loads the model. | 200 |
| `/ready` | Database reachable, migrations at head (production), audit table append-only | 200 with `ready: true`, `migrations: at-head`, `audit_guard: present` |
| `/model/status` | Model state and provenance, `checks.artifact_integrity`, `checks.feature_schema`, threshold policy, `recent_inference` (success rate, p50/p95 latency over the last 200 attempts) | `prediction_ready: true`; `clinically_ready` is always false |
| `/metrics` | Prometheus counters, including `sepsis_audit_log_failures_total` | — |

There are no alert thresholds on `recent_inference.success_rate` or latency
yet. They are set with the pilot criteria.

## 3. Degraded operation

| Condition | What users see | What to do |
|---|---|---|
| Model absent, tampered or incompatible | `/predict*` returns 503 with `model_state`; the UI shows "No risk calculated". Rule-based scores (`/score`) still work. | Restore the verified artifact (§5) |
| Model fails while scoring | 503 "No risk calculated … not a low-risk result"; monitor patients show "No risk calculated" | Check the logs for `error_class`; roll back if it is new |
| Audit store unavailable | Results are withheld (503 / `risk_unavailable`) | Restore the database; results resume by themselves |
| Database down | `/ready` returns 503; the ALB stops routing | Database recovery |

"No risk calculated" is never shown as, or treated as, a low risk.

## 4. Logs

- **Request log:** `sepsis_vitals.requests`, one JSON line per request (request id, route template, status, latency; for inference: outcome, model version, error class). It contains no identifiers.
- **Access audit:** `HIPAA_AUDIT` lines plus the `audit_log` table.
- **Correlation:** every response carries `X-Request-ID`, and the prediction record stores the same `request_id`.

## 5. Rollback

- **Application:** redeploy the previous task definition revision:
  `aws ecs update-service --cluster <c> --service <s> --task-definition <family>:<previous>`.
  Versions before 006 work with the 006 schema (they never update or delete prediction records) but write no seals or request ids. **Do not downgrade the database to roll back the application.**
- **Database:** `alembic downgrade e5f6a7b8c9d0` removes 006. That deletes the request ids and seals of every record written since; back up first. Use it only with approval.
- **Model:** replace the artifact directory with the previous verified one (manifest included) and restart. The API refuses an artifact whose checksums do not match.
- **Pages:** publishing is manual. To withdraw the demo, disable Pages in the repository settings (owner action).

Rollback has not been rehearsed. A drill is a pilot gate (G8).

## 6. Incident response

See `docs/pilot/security_checklist.md` (incident process).
