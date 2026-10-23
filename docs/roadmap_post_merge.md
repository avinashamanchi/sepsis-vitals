# Post-merge roadmap

**Status:** plan, not a commitment. The software is not approved for
clinical use, and nothing below claims pilot or hospital readiness. Each
stage ends at a gate, and the next stage starts only after that gate has
been signed off (`docs/pilot/README.md`).

Ordering principle: decisions and approvals that block everything else come
first. Engineering work runs in parallel only where it does not depend on
them.

## Stage 0: immediately after the merge (owner actions plus one engineering run)

| # | Action | Owner | Done when |
|---|---|---|---|
| 0.1 | Revoke the two GitHub credentials in `.claude/settings.local.json` (`gho_…` fingerprint `d7fa3d690ec4`; `ghp_…` fingerprint `90661b0dd48c`); then remove allow-rules 18, 20, 76, 81, 82, 83 | Project owner | Both tokens are absent from GitHub's token and authorized-app lists; the rules are removed |
| 0.2 | Run "Candidate-complete reports" with the defaults (`workflow_dispatch` on `main`): 20,000 patients, 5 CV folds, 1,000 bootstrap replicates, all 5 backends | Engineering, with owner approval to run | The artifact is reviewed; every report is labelled synthetic development evidence; committed by PR only after review |
| 0.3 | Apply branch protection: the single-maintainer variant (see the PR discussion), or the original after adding a second reviewer | Project owner | `gh api …/branches/main/protection` returns the rule |
| 0.4 | Confirm GitHub account 2FA; enable private vulnerability reporting | Project owner | Settings screenshots or API confirmation |

## Stage 1: clinical definition (blocks the data request and retraining)

Decisions R1–R6 and R10 with the clinical lead, statistician and nursing
lead (`docs/pilot/clinical_requirements.md`):

1. **Intended use and population (R1):**
   - setting, users, the decision supported;
   - silent mode first;
   - inclusion and exclusion criteria.
2. **Labelling standard (R2):**
   - sepsis definition, for example Sepsis-3 with the SOFA and infection criteria and the window;
   - adjudication (`compliance/adjudication_protocol.md`).
3. **Prediction horizon (R3)**, and how observations after onset are handled.
4. **Data requirements (R6):**
   - fields and units;
   - minimum frequency;
   - maximum age of the newest observation;
   - whether labs are required.
5. **Operating point (R4):** the target sensitivity, PPV or alert burden, and how the rule-based scores and the model combine.
6. **Alert routing and escalation (R5)**, including the deterioration rule and the failure behaviour (R7).

**Gate G1:** intended use frozen and signed. Output: a locked target
specification that the training code can take as configuration.

## Stage 2: approved representative data

1. Write the data request from the Stage 1 specification: cohort, fields, period, de-identification method.
2. Approvals: the project owner's written approval for real or de-identified data (E6), a data-use agreement, ethics approval, and a DPIA.
3. Secure environment: storage, access control and audit for the extract. **No PHI in the repository, CI, logs or reports.**
4. Data-quality report on the extract (gate G2): completeness, units, timestamps, outcome ascertainment.

**Gate G2:** data readiness passes.

## Stage 3: retrain (the current model behaves backwards on vital signs)

1. Retrain with `--label-mode onset_within_horizon` and the approved horizon, on the training split only. Use patient-level and temporal splits; site held out if possible.
2. Add behaviour checks to the evaluation gates. On held-out data, a recorded fever or tachycardia must not lower the output (the attribution probes in `sepsis_vitals.ml.model_evaluation`). Check monotonicity for the core vital signs that the clinical lead specifies.
3. Recalibrate on the validation split, and ship an uncertainty statement only if it is valid (for example, a conformal predictor calibrated on real data).
4. Replace the committed artifact only with separate approval. Give it a new manifest and validation status, and archive the old one (`retrain.py --replace`).

**Gate:** the model card is updated, and the behaviour checks pass.

## Stage 4: retrospective validation, then silent mode

1. **Retrospective validation (gates G3–G5):** a locked analysis plan before the results are seen (TRIPOD+AI), covering:
   - discrimination and calibration with CIs;
   - subgroups;
   - decision curve;
   - alert burden at the agreed operating point;
   - missing-data handling;
   - failure-mode review.
   Report retrospective-real results separately from synthetic ones.
2. **Silent-mode study (gate G6)**, under an approved protocol:
   - outputs are recorded and never shown;
   - prespecified stop criteria (`docs/pilot/pilot_criteria.md`, X1–X6);
   - interim review.
3. **Human-factors evaluation (gate G7)** before any display is considered.

## Stage 5: security, privacy and operations (in parallel with Stages 1–4)

| # | Exercise | Done when |
|---|---|---|
| 5.1 | Enforce application MFA for clinical roles (`SEPSIS_MFA_REQUIRED_ROLES`, after `admin_cli mfa-status`) | Every clinical role is enrolled and enforcement is on in the test environment |
| 5.2 | Pseudonymous `patient_id` enforced at the interface; decide whether to encrypt `prediction_records.patient_id` | Validation and tests |
| 5.3 | Retention periods set (privacy officer); privileged, audited purge procedure for append-only records | Policy signed; procedure tested on synthetic data |
| 5.4 | Penetration test and threat-model review | Findings closed or accepted |
| 5.5 | Non-production AWS environment: `terraform plan` (approval), apply (separate approval), pre-serve migration with `scripts/ecs_migrate.py --execute` | Health checks green; `/ready` gated |
| 5.6 | Rollback drill: previous task-definition revision; model artifact restore | Timed and recorded in the runbook |
| 5.7 | Restore drill: an RDS point-in-time restore into an isolated instance; `audit_records verify` on the restored data | All records intact |
| 5.8 | Incident tabletop: suspected PHI exposure and audit-store outage | Owners named; procedure updated |
| 5.9 | Alerting on `recent_inference.success_rate`, latency and `sepsis_audit_log_failures_total` | Thresholds set with operations |

**Gate G8:** safety and operations, signed.

## Out of scope until every gate is passed

Deployment to production, clinical alerts, showing outputs to clinicians,
and any statement of pilot or hospital readiness.
