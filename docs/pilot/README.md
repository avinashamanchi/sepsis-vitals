# Pilot approval package

**The software is not approved for clinical use.** This package collects
what the reviewers need to decide whether a controlled, silent-mode pilot
may be prepared. None of these documents has been signed.

| Document | Purpose |
|---|---|
| [clinical_requirements.md](clinical_requirements.md) | Decisions R1–R10 that clinicians must make (intended use, label, horizon, thresholds, alerting, data, failure behaviour, display, criteria, emergency access) |
| [model_card.md](model_card.md) | What the committed model is, its synthetic-only evidence, and its failure modes |
| [../../reports/committed_model_evaluation.md](../../reports/committed_model_evaluation.md) | Evaluation report: cohort, label, leakage, missingness, AUROC/AUPRC with CIs, calibration, decision curve, subgroups, alert burden, failure modes, with synthetic, retrospective and prospective evidence kept separate |
| [data_provenance_privacy.md](data_provenance_privacy.md) | Data used so far, data the system processes, PHI controls and gaps |
| [security_checklist.md](security_checklist.md) | Access control, encryption, supply chain, retention (draft), incident process (draft) |
| [runbook.md](runbook.md) | Deploy, pre-serve migration, health checks, degraded operation, logs, rollback |
| [pilot_criteria.md](pilot_criteria.md) | Entry, success and stop criteria (structure; the clinical values are set by the owners) |
| [risk_register.md](risk_register.md) | Risks with severity, likelihood, mitigation, owner and acceptance criteria |
| [branch_protection_main.json](branch_protection_main.json) | Draft branch protection (not applied) |
| [../inference_contract.md](../inference_contract.md) | Request validation, outputs, audit record, failure behaviour |
| [../current_state.md](../current_state.md) | The state before Phase 1 |

## Sign-off required before a pilot

| Role | Signs off on | Status |
|---|---|---|
| Clinical lead | R1–R9, the model card, the evaluation on approved data, pilot criteria | Not started |
| Study statistician | Analysis plan, evaluation, criteria S1–S5 | Not started |
| Nursing lead | Alert routing and burden (R4, R5, S3) | Not started |
| Privacy officer | DPIA, retention, data-use agreement, PHI controls | Not started |
| Security lead | Security checklist, penetration test | Not started |
| Operations lead | Runbook, rollback and restore drills | Not started |
| Ethics committee or IRB | Protocol | Not started |
| Regulatory counsel | Intended use and regulatory pathway | Not started |
| Project owner | Use of real patient data (E6), deployment, repository settings | Not started |
