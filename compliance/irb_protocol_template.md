# IRB Protocol Template — Sepsis Vitals

## 1. Study Title

Validation of a Vitals-Only Sepsis Early Warning System in Low-Resource District Hospitals

## 2. Principal Investigator

[Name, Credentials, Affiliation]

## 3. Study Objectives

### Primary
Evaluate the sensitivity and specificity of the Sepsis Vitals scoring system (qSOFA, partial SIRS, NEWS2-style, UVA-style, Shock Index) for detecting sepsis in adult patients presenting to district hospitals in low- and middle-income countries (LMIC).

### Secondary
- Assess alert fatigue burden (alerts per 100 encounters)
- Measure time-to-treatment improvement vs. standard of care
- Evaluate subgroup performance across age, sex, and site

## 4. Study Design

Prospective, multi-site observational cohort study with a pre/post intervention design.

- **Phase 1 (3 months):** Baseline data collection — standard of care only
- **Phase 2 (6 months):** Intervention — Sepsis Vitals deployed alongside standard care
- **Phase 3 (3 months):** Full integration with clinical workflows

## 5. Study Population

### Inclusion Criteria
- Adults aged >= 18 years
- Presenting to emergency or inpatient wards
- At least 3 of 6 core vitals recorded (temperature, heart rate, respiratory rate, systolic BP, SpO2, GCS)

### Exclusion Criteria
- Patients with DNR/comfort-care-only orders
- Incomplete vital sign sets (< 3 vitals recorded)
- Pediatric patients (< 18 years) unless site-specific pediatric module is validated

## 6. Data Collection

### Core Vitals (Minimum Dataset)
| Vital | Unit | Collection Frequency |
|-------|------|---------------------|
| Temperature | °C | Every 4 hours minimum |
| Heart Rate | bpm | Every 4 hours minimum |
| Respiratory Rate | /min | Every 4 hours minimum |
| Systolic BP | mmHg | Every 4 hours minimum |
| SpO2 | % | Every 4 hours minimum |
| GCS | /15 | Every 8 hours minimum |

### Outcome Variables
- Sepsis diagnosis (Sepsis-3 criteria, adjudicated)
- ICU admission
- 28-day mortality
- Time from first abnormal vital to antibiotic administration

## 7. Ethical Considerations

### Informed Consent
- Waiver of individual consent requested (minimal risk, standard-of-care vitals)
- Site-level consent from hospital administration
- Patient notification posters in local languages (English, Swahili)

### Data Protection
- All patient identifiers encrypted with AES-256-GCM
- Data stored on encrypted servers with access logging
- Compliant with local data protection laws, HIPAA BAA, and GDPR DPA where applicable
- See `compliance/data_protection_agreement.md`

### Risk Minimization
- System provides advisory alerts only — all clinical decisions remain with treating clinician
- Alert fatigue monitored continuously (see `monitoring/metrics.py`)
- Automatic escalation suppression if override rate exceeds 70%

## 8. Statistical Analysis Plan

### Sample Size
- Minimum 200 sepsis cases per site for adequate subgroup analysis
- Target enrollment: 4,000 encounters per site per year
- Expected sepsis prevalence: 10-20%

### Primary Analysis
- Sensitivity, specificity, PPV, NPV with 95% confidence intervals
- AUROC for composite risk score
- Subgroup analysis by age group, sex, and site

### Fairness Analysis
- Equalized odds assessment across demographic subgroups
- Calibration curves per subgroup (see `ml/fairness.py`)
- Conformal prediction intervals for uncertainty quantification

## 9. Data Safety Monitoring

- Monthly review of alert fatigue metrics
- Quarterly interim analysis by independent DSMB
- Stopping rules: > 20% absolute difference in mortality between arms

## 10. Timeline

| Phase | Duration | Activities |
|-------|----------|-----------|
| Setup | 2 months | IRB approval, site training, system deployment |
| Phase 1 | 3 months | Baseline data collection |
| Phase 2 | 6 months | Intervention period |
| Phase 3 | 3 months | Full integration |
| Analysis | 2 months | Statistical analysis, manuscript preparation |

## 11. Budget

See `health_economics/model.py` for cost-effectiveness projections.

## 12. References

1. Seymour CW et al. Assessment of clinical criteria for sepsis. JAMA 2016;315(8):762-774.
2. Bone RC et al. Definitions for sepsis and organ failure. Chest 1992;101(6):1644-1655.
3. Royal College of Physicians. National Early Warning Score (NEWS) 2. 2017.
4. Kruisselbrink R et al. Modified Early Warning Score (MEWS) recognizes patients at risk. PLOS ONE 2016.
