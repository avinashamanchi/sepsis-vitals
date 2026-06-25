# Sepsis-Vitals: Complete ML Pipeline, Autonomous Prediction & UI Overhaul

**Date:** 2026-06-25
**Status:** Approved (Rev 2 — clinical corrections applied)
**Scope:** Full-stack rebuild of ML training, prediction engine, simulator, and frontend

## Overview

Transform sepsis-vitals from a single-model, on-demand prediction tool into a production-grade autonomous clinical monitoring platform with:

1. Real MIMIC-IV data ingestion with Sepsis-3 clinical labeling (not billing codes)
2. Single heavily-regularized model for 100-patient demo; ensemble architecture ready for scale
3. Event-driven continuous monitoring with clinically meaningful deterioration tracking
4. Real patient replay and synthetic ward simulation for demos
5. Clinical-grade monitoring dashboard, enhanced patient detail, and full prediction workflow

Built in 5 sequential layers, each independently testable.

---

## Layer 1: Data Pipeline

### Goal
Parse both MIMIC-IV data sources into a unified training DataFrame with clinically valid Sepsis-3 labels and time-window-binned observations.

### Critical Design Decision: Sepsis-3 Labeling (Not Billing Codes)

ICD codes (A40/A41/R65.2) are retrospective billing artifacts assigned at discharge. Using them as labels introduces future information leakage — the model learns to predict what a coder decided after the fact, not what's happening clinically in real time.

**Sepsis-3 operational definition (Singer et al., JAMA 2016):**
1. **Suspected infection:** Antibiotic administration within ±72h of a blood/body fluid culture order
2. **Organ dysfunction:** SOFA score increase ≥ 2 from baseline within ±48h of suspected infection onset

**MIMIC-IV Demo data available for this:**
- `hosp/prescriptions.csv.gz` + `hosp/emar.csv.gz` — antibiotic administration timestamps
- `hosp/microbiologyevents.csv.gz` — culture order timestamps
- `icu/chartevents.csv.gz` — vitals for SOFA component calculation (GCS, MAP, vasopressors)
- `hosp/labevents.csv.gz` — creatinine, bilirubin, platelets for SOFA components
- `icu/inputevents.csv.gz` — vasopressor administration for cardiovascular SOFA

**Sepsis onset time (t_sepsis):** The earlier of antibiotic start or culture order, provided the other occurs within 72h. This is the clinical ground truth timestamp — every observation before t_sepsis is labeled 0, every observation at/after t_sepsis is labeled 1.

**Fallback for incomplete data:** If antibiotic/culture data is missing for a patient, fall back to ICD code presence as a *stay-level* label (sepsis at some point during stay), but mark these patients with `label_source: "icd_fallback"` so they can be excluded from time-sensitive analysis.

### New: `src/sepsis_vitals/ml/sepsis3_labeler.py`

Dedicated Sepsis-3 label derivation module:
- `compute_sofa_score(vitals, labs, vasopressors, ventilation)` — 6-component SOFA (respiratory, coagulation, liver, cardiovascular, CNS, renal)
- `find_suspected_infection(prescriptions, microbiology)` — antibiotic + culture co-occurrence within 72h window
- `derive_sepsis_onset(sofa_series, infection_time)` — t_sepsis = time of SOFA ≥ 2 increase within ±48h of suspected infection
- `label_observations(observations, t_sepsis)` — assigns per-observation binary labels relative to onset time

### New: `src/sepsis_vitals/ml/fhir_loader.py`

Parses downloaded FHIR NDJSON.gz files using **streaming/chunked parsing** to avoid OOM on 668K chartevent records:

- Uses `ijson` or line-by-line gzip iteration (each NDJSON line is one JSON object — no need for full-file parse)
- Processes in configurable chunks (default 10,000 records) with intermediate DataFrame aggregation
- Memory ceiling: ~500MB peak regardless of file size

Extracts:
- **Patient demographics** from `MimicPatient.ndjson.gz` — age (from birthDate), sex, race/ethnicity via US Core extensions
- **Vitals** from `MimicObservationChartevents.ndjson.gz` — maps MIMIC itemid codes using `CHART_VITALS` mapping
- **Labs** from `MimicObservationLabevents.ndjson.gz` — lactate, WBC, procalcitonin, plus SOFA components (creatinine, bilirubin, platelets, PaO2/FiO2)
- **Medications** from `MimicMedicationAdministration.ndjson.gz` — antibiotic administration for Sepsis-3 suspected infection
- **Microbiology** from `MimicObservationMicroTest.ndjson.gz` — culture orders for Sepsis-3
- **ICU encounters** from `MimicEncounterICU.ndjson.gz` — links to ICU stays
- **Comorbidities** from `MimicCondition.ndjson.gz` — ICD-10 prefix matching

Output: DataFrame with Sepsis-3 labels via `sepsis3_labeler.py`, not billing codes.

### Modified: `src/sepsis_vitals/ml/mimic_loader.py`

Changes:
- Update `_validate_paths()` to handle the demo directory layout
- Add `from_demo()` classmethod pointing to `physionet.org/files/mimic-iv-demo/2.2/`
- Replace `derive_sepsis_labels()` ICD-based method with call to `sepsis3_labeler.py`
- Add SOFA component loading from labevents (creatinine, bilirubin, platelets) and inputevents (vasopressors)

### New: `src/sepsis_vitals/ml/data_unifier.py`

Merges outputs from both loaders into a single deduplicated DataFrame:
- Patient ID normalization: FHIR UUIDs mapped to MIMIC subject_id integers via identifier field
- **Time-window binning**: Observations binned into 1-hour epochs per patient. Within each epoch, vitals are aggregated (median for continuous values, max for GCS). Forward-fill from previous epoch for missing vitals. This handles the clinical reality that HR logged at 10:01 and BP at 10:45 belong to the same clinical assessment window.
- Feature column alignment to match the 62-feature schema
- Deduplication by (patient_id, epoch_timestamp) after binning

---

## Layer 2: ML Model

### Goal
Train a clinically appropriate model on MIMIC-IV data. Single heavily-regularized model for the 100-patient demo dataset; ensemble architecture code ready to activate when full MIMIC-IV credentials arrive (40,000+ patients).

### Critical Design Decision: No Ensemble on 100 Patients

Stacking 4 correlated tree models (GBM, XGBoost, LightGBM, RF) on ~100 patients with ~10-15 actual sepsis cases is statistically unsound. The base models will be highly correlated (all tree-based), and the stacker will aggressively memorize noise. Cross-validation on this sample size produces unreliable out-of-fold estimates.

**Demo dataset strategy:**
- Train a single **LightGBM** with heavy regularization: `max_depth=3`, `num_leaves=8`, `min_data_in_leaf=20`, `lambda_l1=1.0`, `lambda_l2=1.0`, `feature_fraction=0.6`
- Also train **Logistic Regression** (L2-regularized) as interpretable baseline
- Select the better model by leave-one-patient-out cross-validation (LOPOCV) — more appropriate than 5-fold CV for tiny datasets
- Report both models' metrics transparently

**Ensemble readiness for full MIMIC-IV (>500 patients):**
- `ensemble.py` module is built and tested but not activated for the demo
- Activation threshold: `if n_patients >= 500: use_ensemble = True`
- When activated: 4 base models + logistic stacker + clinical weighting (as originally designed)

### Critical Design Decision: No Real-Time SHAP

Computing SHAP TreeExplainer values for every prediction on an edge device will pin CPU. Instead:
- **Training time:** Compute and store global SHAP feature importance values (one-time cost)
- **Inference time:** Use pre-computed model feature importance (already available from tree-based models via `.feature_importances_`). This is what the current system already does.
- **Per-patient explanation:** Only compute on-demand when a clinician clicks "Explain this prediction" in the UI (not on every 60-second cycle). Use cached SHAP explainer, compute for single sample.

### Critical Design Decision: ≥99% Specificity for Continuous Monitoring

95% specificity = 5% FPR. At 100 vitals checks/day in an ICU, that's 5 false alarms per patient per day. Nurses will mute the app.

Operating point selection:
- **Continuous monitoring mode:** Threshold tuned for ≥99% specificity (~1% FPR). Accept lower sensitivity — the goal is "when this alerts, pay attention." Approximately 1 false alarm per patient per 4 days.
- **On-demand prediction mode** (manual vitals entry): Threshold at ≥95% specificity. Higher sensitivity acceptable since the clinician is already engaged and can assess context.
- Both thresholds stored in model metadata and applied at prediction time based on mode.

### New: `src/sepsis_vitals/ml/ensemble.py`

Built and tested, activated only when dataset is large enough:

```
class SepsisEnsemble:
    base_models: list[BaseModel]      # 4 trained models
    stacker: LogisticRegression        # meta-learner
    calibrator: CalibratedClassifierCV # Platt scaling
    clinical_weights: np.ndarray       # sensitivity-based weights
    imputation_medians: dict           # from training data
    feature_names: list[str]           # 62 features
    MIN_PATIENTS_FOR_ENSEMBLE = 500    # activation threshold

    def predict(features: np.ndarray) -> float
    def predict_with_uncertainty(features) -> (prob, ci_lower, ci_upper)
    def explain(features) -> list[FeatureImportance]  # pre-computed importance, not live SHAP
```

### Modified: `src/sepsis_vitals/ml/predictor.py`

- Auto-detect model type from metadata: `"ensemble"` vs `"single"`
- Dual operating points: `mode="continuous"` (99% spec) vs `mode="on_demand"` (95% spec)
- Feature importance from pre-computed values, on-demand SHAP only when explicitly requested
- Confidence intervals: bootstrap-based for single model, inter-model disagreement for ensemble

### Modified: `src/sepsis_vitals/train.py` / `retrain.py`

```bash
python -m sepsis_vitals.train --data-source mimic-demo --output models
python -m sepsis_vitals.train --data-source mimic-demo --ensemble --output models  # errors if <500 patients
python -m sepsis_vitals.train --data-source synthetic --patients 20000  # existing behavior
```

### Validation Metrics

- AUROC, AUPRC, Brier score
- Sensitivity at 95% specificity (on-demand threshold)
- Sensitivity at 99% specificity (continuous monitoring threshold)
- PPV/NPV at both operating points
- LOPOCV results (mean ± std across patient folds)
- Fairness: AUROC stratified by age bracket and sex

---

## Layer 3: Autonomous Prediction Engine

### Goal
Event-driven monitoring that triggers predictions on data arrival, tracks deterioration over clinically meaningful windows, and escalates alerts.

### Critical Design Decision: Event-Driven, Not Polling

Sepsis is a slow-burn physiological cascade developing over hours. Vitals update every 15-60 minutes; labs every 12-24 hours. A blind 60-second polling loop wastes compute and produces redundant predictions.

**Architecture: Data-arrival triggered predictions.**
- `VitalsIngester` is the single entry point for all data
- When new vitals/labs arrive for a patient, the ingester triggers a prediction immediately
- No background polling loop. No timer. Predictions happen exactly when new clinical data is available.
- Minimum re-prediction interval: 5 minutes (debounce). If vitals arrive faster (e.g., continuous telemetry), buffer and predict on the epoch boundary.

### Critical Design Decision: 2-Hour Deterioration Window

Escalating an alert because risk jumped 0.1 over 3 minutes is clinically meaningless — a patient coughing or shifting position will alter HR/RR enough to trigger it.

**Deterioration tracking:**
- Compute risk trend over a **2-hour rolling window** (not 3 readings)
- Escalation criteria: risk probability increased ≥ 0.15 sustained over the 2-hour window AND current risk is above the on-demand threshold (0.50)
- De-escalation: risk decreased ≥ 0.15 sustained over 2 hours → send recovery notification
- Single-reading spikes are noted but not escalated

### Critical Design Decision: Redis/PostgreSQL, Not SQLite

The existing architecture uses PostgreSQL for persistent state and Redis for ephemeral/cache. SQLite was an earlier prototype decision. The monitoring engine must use the established infrastructure:
- **PostgreSQL:** Patient registry state, prediction history, alert audit trail (immutable)
- **Redis:** Current patient vitals cache (fast read/write), active monitoring set, debounce timers
- **No SQLite** in the monitoring path. `state_store.py` SQLite remains only for standalone/demo mode fallback.

### New: `src/sepsis_vitals/ml/monitor.py`

**`PatientRegistry`**
- In-memory registry backed by Redis (current state) and PostgreSQL (history)
- Per-patient state: current vitals, vitals history (2-hour rolling window), last prediction time, risk trajectory, alert state
- Async-safe access via Redis atomic operations

**`VitalsIngester`**
- Unified intake for all data sources:
  - `ingest_single(patient_id, vitals, demographics?, comorbidities?)` — from UI form
  - `ingest_batch(records[])` — from batch upload
  - `ingest_fhir(observation_bundle)` — from FHIR webhook
- All paths normalize to internal format, register patient if new
- **Triggers prediction immediately** on data arrival (with 5-minute debounce)
- Time-window bins incoming vitals into 1-hour epochs matching training data format

**`DeteriorationTracker`**
- Maintains 2-hour rolling window of predictions per patient
- Computes trend: linear slope of risk probability over the window
- Escalation: sustained increase ≥ 0.15 over 2h + current risk above threshold
- De-escalation: sustained decrease ≥ 0.15 over 2h → recovery notification
- Alert state machine: `normal -> elevated -> escalated -> critical` (no skipping levels without sustained data)

### Modified: `src/sepsis_vitals/api.py`

New endpoints:
- `POST /monitor/register` — Register patient for monitoring
- `DELETE /monitor/{patient_id}` — Remove from monitoring
- `GET /monitor/status` — List monitored patients with current risk, trend, last update
- `POST /predict` — Also feeds vitals into registry (smart rerun)
- `POST /predict/batch` — Registers all patients for monitoring

No background loop started in lifespan — predictions are event-driven from ingestion.

### Modified: `src/sepsis_vitals/realtime/websocket.py`

Enhanced alert messages:
- `alert_type`: `"new_risk"` | `"escalation"` | `"deterioration"` | `"recovery"`
- `previous_risk_level`, `risk_delta`, `window_hours` for context
- `deterioration_rate`: risk increase per hour (computed from 2h window slope)
- `patient_update` message type for vitals/risk refreshes (separate from alerts)

### Modified: `src/sepsis_vitals/fhir/router.py`

Wire FHIR endpoint to ingester:
- Parse FHIR Observation resources -> extract vitals/labs via LOINC/MIMIC code mapping
- Feed into `VitalsIngester.ingest_fhir()`
- Return FHIR OperationOutcome response

---

## Layer 4: Simulator

### Goal
Replay real MIMIC-IV cases and generate synthetic wards for demos. Both feed through the same prediction pipeline as production data.

### Critical Design Decision: Synchronized Simulation Clock

If the simulator runs 24h of data in 2 minutes but predictions are event-driven, the simulator must control the pacing. Each observation is emitted at the correct real-time interval (scaled by speed multiplier), and the prediction engine processes it on arrival. No clock desync because there's no independent timer — the simulator IS the data source.

**Implementation:** The simulator emits observations via `VitalsIngester.ingest_single()` with real timestamps. The ingester triggers prediction immediately. The speed multiplier only controls the delay between emissions (e.g., at 720x speed, observations that were 1 hour apart in real time are emitted ~5 seconds apart).

### New: `src/sepsis_vitals/ml/simulator.py`

**`CaseReplay`**
- Loads a MIMIC-IV patient's full ICU stay from `case_library.py`
- Replays vitals chronologically at configurable speed (default: 720x, so 24h -> 2 min)
- Each observation emitted via `VitalsIngester.ingest_single()` at time-scaled intervals
- Frontend sees real vitals arriving, risk updating, alerts firing
- Metadata: patient subject_id, source institution, ICU stay info, duration, current position

**`WardSimulator`**
- Generates configurable ward using `synthetic_data.py`
- Default: 8-12 patients. Mix: ~2 sepsis, ~3 sick confounders, ~5 stable
- Async loop emitting observations at accelerated pace
- One patient scripted to deteriorate low->critical (guaranteed escalation demo)
- All observations flow through `VitalsIngester` -> prediction -> WebSocket

Both simulators feed through the real pipeline. No separate demo path.

### New: `src/sepsis_vitals/ml/case_library.py`

Indexes MIMIC-IV demo data for quick lookup:
- Builds lightweight SQLite index (read-only reference data, appropriate for SQLite)
- Fields: subject_id, age, sex, sepsis_onset_time (from Sepsis-3 labeler), icu_los_hours, n_observations
- `get_sepsis_cases()` — patients with Sepsis-3 positive labels
- `get_case(subject_id)` — full vitals timeline for replay
- `get_random_case(sepsis=True)` — random matching case

### Modified: `src/sepsis_vitals/api.py`

Simulator endpoints (gated behind `ENABLE_SIMULATOR=true`):
- `POST /simulator/replay` — Start MIMIC-IV patient replay. Accepts `{subject_id: int | "random", speed: float, sepsis_only: bool}`. Returns session_id.
- `POST /simulator/ward` — Start ward simulation. Accepts `{n_patients: int, speed: float, sepsis_count: int}`. Returns session_id.
- `DELETE /simulator/{session_id}` — Stop simulation
- `GET /simulator/sessions` — List active simulations with progress
- `GET /simulator/cases` — List available cases for replay selection

---

## Layer 5: Frontend UI

### Goal
Three new/overhauled pages connected to the autonomous prediction engine, designed for clinical workflows.

### Critical Design Decision: Fixed Patient Order (No Auto-Sort)

Auto-sorting patients so critical ones float to top is a notorious clinical UX anti-pattern. If a nurse goes to click Patient 4 and Patient 8's risk score ticks up at that millisecond, the grid re-sorts and the nurse clicks the wrong patient. This is a known cause of medication errors.

**Sort order:** Fixed by bed/registration order. Never re-sorts while the page is open. Risk level indicated by:
- Card border color: green (low), amber (moderate), red (high), pulsing red (critical)
- Risk badge prominently displayed on each card
- Optional "Sort by Risk" button that requires explicit user action (not automatic)

### Critical Design Decision: Clinically Meaningful Sparklines

Sparklines showing the last 12 data points are only useful if those points span a clinically relevant time window. At 60-second intervals, 12 points = 12 minutes = useless flat line for sepsis.

**Sparkline window:** Last 24 hours of risk scores (or the full monitoring duration if < 24h). This shows the actual sepsis trajectory — gradual onset over hours, which is the pattern clinicians need to see.

### New: `frontend/src/pages/Monitor.tsx`

Real-time ward monitoring — the central station display:

- **Patient grid**: Cards for all monitored patients, **fixed order by bed/registration**. Each card shows:
  - Patient ID and demographics
  - Current risk badge (RiskBadge component) with level
  - **24-hour sparkline** of risk scores (Recharts LineChart, minimal, no axes)
  - Latest vitals snapshot: HR, Temp, SBP, SpO2 with abnormal values highlighted in red
  - Time since last observation ("2m ago", "15m ago")
  - Trend arrow: improving (green down-arrow), stable (gray horizontal), worsening (red up-arrow)
- **Risk indication via borders/badges** — not sort order. Critical patients get pulsing red border.
- **Explicit sort button**: "Sort by Risk" button for when the clinician actively wants to triage. Requires click, doesn't auto-update.
- **Live WebSocket updates**: Vitals and risk update in-place without reordering.
- **Simulation controls** (when simulator active): Control bar at top — play/pause, speed slider (1x-1000x), session info, stop button.
- **Click-through**: Patient card navigates to `/patients/:id`

### Overhauled: `frontend/src/pages/PatientDetail.tsx`

Enhanced from basic charts to comprehensive clinical view:

- **Risk timeline**: Full risk probability over time as area chart with confidence interval band (shaded). Horizontal threshold lines at 0.25/0.50/0.75 labeled moderate/high/critical. **24-hour default window**, expandable to full stay.
- **Clinical scores panel**: qSOFA, NEWS2, SIRS, Shock Index displayed as colored number cards with label and interpretation. Updated per observation.
- **Feature importance chart**: Per-prediction feature importance as horizontal bar chart (pre-computed model importance, not live SHAP). "Explain in detail" button triggers on-demand SHAP computation for that single prediction.
- **Vitals history**: Multi-line chart with toggleable vitals. Normal range bands as subtle colored fills. 24-hour default window.
- **Alert history timeline**: Chronological list of alerts for this patient with acknowledgment status, clinical notes, response time.
- **Deterioration indicator**: Trend arrow with rate text ("Worsening: +12% risk/hour over last 2h")

### Overhauled: `frontend/src/pages/Predict.tsx`

Full prediction workflow with all model inputs exposed:

- **Demographics section**: Age (number input), Sex (select: M/F). Pre-filled if patient exists in system.
- **Comorbidities section**: 5 checkboxes — Hypertension, Diabetes, CKD, COPD, Heart Failure.
- **Vitals section**: Existing 9-vital form (temperature through procalcitonin), unchanged.
- **Clinical scores display**: After prediction, show computed qSOFA, NEWS2, SIRS, Shock Index in a row of cards alongside the ML probability.
- **Dual threshold display**: Show risk at both operating points — "Continuous monitoring: LOW (below 99% spec threshold)" / "Clinical assessment: MODERATE (above 95% spec threshold)". This helps clinicians understand why the continuous monitor might not alert but a manual assessment flags concern.
- **Historical comparison**: If patient_id has prior predictions, show mini risk trajectory. "Risk increased from 32% -> 48% since last assessment 4 hours ago."
- **Auto-monitor toggle**: "Add to continuous monitoring" checkbox.

### New: `frontend/src/components/SimulatorPanel.tsx`

Floating control panel (bottom-right, collapsible):
- **Replay tab**: Dropdown of MIMIC-IV cases (labeled sepsis/no-sepsis, age, sex, ICU LOS). "Start Replay" button.
- **Ward sim tab**: Patient count slider (4-20), speed multiplier (1x-1000x), sepsis count. "Start Ward" button.
- **Active sessions**: List with play/pause/stop, progress bar for replays.
- **Visibility**: Only rendered when backend reports `simulator_enabled: true`.

### Modified: `frontend/src/stores/useStore.ts`

New state:
- `monitoredPatients: Record<string, MonitoredPatient>` — patient_id -> current state (risk, vitals, trend, last_update)
- `simulatorSessions: SimSession[]` — active simulation sessions
- `updatePatientRisk(id, prediction)` — from WebSocket, updates in-place without reordering
- `setMonitoredPatients(patients)` — bulk update from REST
- `addSimSession(session)` / `removeSimSession(id)` — simulator state

### Modified: `frontend/src/hooks/useWebSocket.ts`

New message types:
- `patient_update` — vitals/risk update for monitored patient -> `updatePatientRisk()`
- `deterioration_alert` — escalation with previous_risk_level, risk_delta, window_hours
- `simulator_event` — simulation progress (position, completion)

### Modified: `frontend/src/App.tsx`

- New route: `/monitor` -> `Monitor` page (lazy loaded)
- Nav entry: Activity icon, "Monitor" label (Sidebar + BottomNav)

### Modified: `frontend/src/types/index.ts`

New types:
- `MonitoredPatient`: { id, risk_level, risk_probability, vitals, trend_direction, last_update, alert_count, bed_number }
- `SimSession`: { id, type: "replay" | "ward", status, progress?, patient_count, started_at }
- `ClinicalScores`: { qsofa, news2, sirs, shock_index }
- `DeteriorationAlert`: extends Alert with previous_risk_level, risk_delta, deterioration_rate, window_hours

---

## Implementation Order

1. **Layer 1: Data Pipeline** — sepsis3_labeler.py, fhir_loader.py (chunked), mimic_loader.py (Sepsis-3), data_unifier.py (time-window binning)
2. **Layer 2: ML Model** — Single regularized LightGBM + LogReg baseline on demo data, ensemble.py (built but gated at 500 patients), dual operating points (99%/95% specificity)
3. **Layer 3: Prediction Engine** — Event-driven monitor.py (no polling), Redis/Postgres state, 2-hour deterioration window, VitalsIngester
4. **Layer 4: Simulator** — case_library.py (Sepsis-3 indexed), simulator.py (clock-synced replay + ward), API endpoints
5. **Layer 5: UI** — Monitor (fixed order, 24h sparklines), PatientDetail (on-demand SHAP), Predict (dual thresholds, comorbidities), SimulatorPanel

---

## Technical Constraints

- **100-patient dataset**: Single regularized model, no ensemble. LOPOCV for evaluation. Ensemble activates at ≥500 patients.
- **Memory budget**: FHIR parser streams line-by-line, ~500MB peak. No full-file JSON parse.
- **CPU budget**: No real-time SHAP. Pre-computed feature importance for routine predictions. On-demand SHAP for single-patient detail view only.
- **Specificity floor**: ≥99% for continuous monitoring (≤1 false alarm per patient per 4 days). ≥95% for on-demand assessment.
- **State persistence**: Redis (current state/cache) + PostgreSQL (history/audit). No SQLite in production monitoring path.
- **HIPAA note**: MIMIC-IV Demo is de-identified Open Access. Production requires full HIPAA compliance.
- **Model disclaimer**: All predictions carry "Research Use Only — Not FDA Cleared" in the UI.
- **Existing stack**: FastAPI + React + Zustand + Recharts. No new frameworks.
- **Demo mode**: Frontend `isDemo` flag for GitHub Pages. New features degrade gracefully.

---

## Revision History

- **Rev 1 (2026-06-25):** Initial design. Approved by user.
- **Rev 2 (2026-06-25):** Clinical corrections applied — 12 issues fixed:
  1. Sepsis-3 labeling replaces ICD billing codes (future leakage fix)
  2. Time-window binning replaces naive deduplication (1-hour epochs)
  3. Streaming FHIR parser replaces in-memory parse (OOM prevention)
  4. Single regularized model replaces 4-model ensemble on 100 patients (overfitting prevention)
  5. Pre-computed feature importance replaces real-time SHAP (CPU budget)
  6. ≥99% specificity for continuous monitoring replaces 95% (alert fatigue prevention)
  7. Event-driven prediction replaces 60-second polling loop (clinical data cadence)
  8. Redis/PostgreSQL replaces SQLite for monitoring state (no container regression)
  9. 2-hour deterioration window replaces 3-reading window (clinical stability)
  10. Synchronized simulation clock via event-driven architecture (no desync possible)
  11. Fixed patient order replaces auto-sort by risk (clinical UX safety)
  12. 24-hour sparklines replace 12-reading sparklines (clinically meaningful trends)
