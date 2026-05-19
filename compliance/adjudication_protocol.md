# Case Adjudication Protocol — Sepsis Vitals

## 1. Purpose

Standardize the process for adjudicating sepsis cases used in model training, validation, and performance reporting. Ensures consistent gold-standard labels across sites.

## 2. Adjudication Committee

- **Minimum 2 independent reviewers** per case
- At least 1 board-certified physician (internal medicine, emergency medicine, or critical care)
- At least 1 clinical nurse specialist with sepsis management experience
- **Tie-breaker:** Third reviewer if disagreement persists after discussion

## 3. Case Selection

### For Model Training
- All cases flagged as sepsis by any scoring method (qSOFA >= 2, SIRS >= 2, NEWS2 >= 5)
- Random sample of 10% of non-flagged cases (to capture false negatives)
- All in-hospital deaths within 48 hours of admission

### For Model Validation
- Consecutive enrollment — no cherry-picking
- Minimum 200 adjudicated sepsis cases per validation site

## 4. Adjudication Criteria

### Sepsis-3 Definition (Primary)
- **Suspected infection:** Clinical documentation of infection OR antimicrobial initiation
- **Organ dysfunction:** SOFA score increase >= 2 from baseline

### Modified Criteria for Low-Resource Settings
Where lab values are unavailable:
- Clinical signs of infection (documented fever, localizing signs)
- Organ dysfunction based on available vitals (hypotension, tachypnea, altered mental status)
- Response to antimicrobial therapy

## 5. Adjudication Categories

| Category | Code | Definition |
|----------|------|------------|
| Definite sepsis | 1 | Meets Sepsis-3 criteria with confirmatory cultures |
| Probable sepsis | 2 | Meets clinical criteria, cultures unavailable or negative |
| Possible sepsis | 3 | Partial criteria met, alternative diagnosis possible |
| Not sepsis | 0 | Alternative diagnosis confirmed |
| Indeterminate | -1 | Insufficient data for classification |

## 6. Process

1. **Case preparation:** De-identified case packets with vitals, clinical notes, lab results (if available), imaging, treatment
2. **Independent review:** Each reviewer classifies independently
3. **Agreement check:** If both reviewers agree → final label
4. **Discordance resolution:** Joint discussion with case review, re-classify
5. **Tie-breaker:** Third reviewer if still discordant
6. **Documentation:** Final label, reviewer IDs, confidence level, time spent

## 7. Quality Metrics

- **Inter-rater reliability:** Cohen's kappa >= 0.75 required
- **Adjudication rate:** >= 95% of selected cases must be adjudicated
- **Turnaround:** Cases adjudicated within 14 days of selection
- **Bias monitoring:** Track adjudication patterns by reviewer, site, demographics

## 8. Training

All adjudicators must complete:
- 2-hour training session on Sepsis-3 criteria
- 10 practice cases with consensus answers
- Annual recertification with 5 test cases (>= 80% agreement required)
