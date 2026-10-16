# Pilot success and stop criteria (structure for sign-off)

**Status:** draft structure. Clinical and statistical values are not set
here; they come from the owners in `docs/pilot/clinical_requirements.md`
(R9). The proposals ⟨…⟩ in `compliance/intended_use_and_validation_plan.md`
(gates G1–G8) are a starting point for that discussion. They are not
decisions.

The criteria apply to a **silent-mode** pilot: outputs are recorded and
nothing is shown to clinicians. Showing outputs would need a separate
decision and approval.

## Entry criteria (all required before the pilot starts)

| # | Criterion | Evidence |
|---|---|---|
| E1 | R1–R10 decided and signed | `docs/pilot/README.md` sign-off table |
| E2 | A model trained and evaluated on approved, representative data (gates G2–G5) | A new model card and evaluation report |
| E3 | Ethics approval, data-use agreement, DPIA | Documents on file |
| E4 | Security items closed: MFA enforced, branch protection applied, penetration test, retention set, incident owners named | `docs/pilot/security_checklist.md` |
| E5 | Deployment rehearsed: pre-serve migration, rollback and restore drill | Runbook sign-off |
| E6 | The project owner explicitly approves use with real patient data | Written approval |

## Success criteria (values set by the owners)

| # | Measure | Source | Target |
|---|---|---|---|
| S1 | Discrimination for the agreed label and horizon | Locked analysis | ⟨set by statistician⟩ |
| S2 | Calibration | Locked analysis | ⟨set⟩ |
| S3 | Alert burden at the agreed operating point (alerts per 100 patient-days) | `prediction_records` | ⟨set by nursing lead⟩ |
| S4 | Lead time before onset | `prediction_records` against adjudicated onset | ⟨set⟩ |
| S5 | Subgroup performance | Locked analysis | ⟨set⟩ |
| S6 | Operational availability: share of eligible observations with a recorded result | `/model/status` `recent_inference`; records | ⟨set⟩ |

## Stop criteria

These integrity and safety conditions stop the pilot as soon as they are
confirmed. They are proposals for sign-off.

| # | Condition |
|---|---|
| X1 | Any released result without a matching sealed `prediction_records` row, or any record that fails `verify_record` |
| X2 | Any confirmed PHI in logs, analytics, error responses or repository files |
| X3 | Any unauthorised access across sites, or a confirmed account compromise |
| X4 | Model output shown to clinicians during a silent-mode pilot |
| X5 | A deployed artifact that differs from the approved manifest |
| X6 | ⟨Clinical:⟩ the alert burden, missed-case rate or subgroup harm signals defined by the clinical lead (G6 stop criteria) |

## Review cadence

To be decided (proposal: weekly operational review, plus an interim
clinical review at a prespecified point).
