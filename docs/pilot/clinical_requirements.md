# Pilot clinical requirements

**Status:** requirements request. Every value marked ⟨…⟩ or "to be decided"
belongs to the named owner and has not been decided. The engineering team
has proposed none of them as a clinical value. Background and the original
blocker list: `compliance/clinical_requirements_needed.md`. Draft intended
use and proposed gates: `compliance/intended_use_and_validation_plan.md`.

A controlled pilot cannot start until requirements R1–R10 are decided and
signed off (see `docs/pilot/README.md`).

| ID | Requirement | Owner | Decision needed | Where it goes in the software |
|---|---|---|---|---|
| R1 | Intended use | Clinical lead, project owner, regulatory counsel | Setting and level of care; users; the decision supported; silent mode (no output shown) or shown; inclusion and exclusion criteria | `PredictionResponse.intended_use`; UI notice; model card |
| R2 | Reference standard and label | Clinical lead, statistician | Sepsis definition (for example Sepsis-3: which SOFA rise, which infection criterion and window); adjudication (`compliance/adjudication_protocol.md`) | `--label-mode onset_within_horizon` and the labeller |
| R3 | Prediction horizon | Clinical lead | Hours before onset that a warning should come; handling of observations after onset | `--horizon-hours` (required, no default) |
| R4 | Operating point and alert rule | Clinical lead, statistician, nursing lead | Target sensitivity, PPV or alert burden; whether the rule-based scores, the model, or both drive alerts; how the two combine | `ml/decision_policy.py` (a new policy id; the digest is recorded with each result) |
| R5 | Alert routing and escalation | Nursing lead, clinical governance | Who receives what, through which channel, the acknowledgement time and escalation path; deterioration rule | `alerts/`, `ml/monitor.py` (`DeteriorationTracker.policy()`) |
| R6 | Data requirements | Clinical lead, site informatics | Required fields and units; minimum observation frequency; **maximum acceptable age of the newest observation**; whether labs are required; handling of a missing age | `schemas.TimeseriesPredictRequest`; `data_quality` warnings; currently a missing age becomes 55 (warned) |
| R7 | Behaviour on failure | Clinical governance | When no risk can be calculated or recorded: confirm the current default ("No risk calculated", result withheld, never shown as low) or specify another; who is notified | `docs/inference_contract.md` §5; runbook |
| R8 | Display | Clinical lead, human factors | Whether a number is shown at all, or only a level; wording of the decision-support notice; attribution display; reference ranges | `frontend/src/pages/Predict.tsx`, `Monitor.tsx` |
| R9 | Pilot success and stop criteria | Clinical lead, statistician, project owner | Values for `docs/pilot/pilot_criteria.md` | Pilot protocol |
| R10 | Emergency access (break-glass) | Clinical governance, privacy officer | Policy (N18) | Currently always 403 |

## What is already fixed, and does not need a clinical decision

- No model output is presented as a diagnosis (`decision_support_notice`, `clinical_use: not-permitted`).
- The model can never lower the rule-based level.
- A failed or unrecorded inference is never shown as a score.
- Every result is traceable: input digest, model, preprocessing, threshold policy, time, request.

## Evidence the decisions depend on

- Retrospective evaluation on approved, representative, de-identified data
  (M7). None exists. The committed model was trained on synthetic data
  whose vital signs do not separate septic from non-septic observations
  (N53, `reports/committed_model_evaluation.md`), so **the model must be
  retrained on approved data before R4 can be decided.**
- A silent-mode prospective protocol (gate G6) with ethics approval.
