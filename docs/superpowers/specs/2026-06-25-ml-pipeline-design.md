# Sepsis-Vitals: Complete ML Pipeline, Autonomous Prediction & UI Overhaul

**Date:** 2026-06-25
**Status:** Approved
**Scope:** Full-stack rebuild of ML training, prediction engine, simulator, and frontend

## Overview

Transform sepsis-vitals from a single-model, on-demand prediction tool into a production-grade autonomous clinical monitoring platform with:

1. Real MIMIC-IV data ingestion (both FHIR NDJSON and relational CSV)
2. Multi-model stacking ensemble with clinical sensitivity weighting
3. Autonomous continuous monitoring with deterioration tracking
4. Real patient replay and synthetic ward simulation for demos
5. Clinical-grade monitoring dashboard, enhanced patient detail, and full prediction workflow

Built in 5 sequential layers, each independently testable.

---

## Layer 1: Data Pipeline

### Goal
Parse both MIMIC-IV data sources into a unified training DataFrame compatible with `trainer.py`.

### New: `src/sepsis_vitals/ml/fhir_loader.py`

Parses downloaded FHIR NDJSON.gz files (100 patients, 668K chartevents, 107K labevents):

- **Patient demographics** from `MimicPatient.ndjson.gz` — age (from birthDate), sex, race/ethnicity via US Core extensions
- **Vitals** from `MimicObservationChartevents.ndjson.gz` — maps MIMIC itemid codes (223761->temperature, 220045->heart_rate, 220210/224690->resp_rate, 220179/220050->sbp, 220180/220051->dbp, 220277->spo2, 223901/223900/220739->gcs, 220052/220181->map) using the same `CHART_VITALS` mapping from `mimic_loader.py`
- **Labs** from `MimicObservationLabevents.ndjson.gz` — lactate (50813), WBC (51265), procalcitonin (50889)
- **Sepsis labels** from `MimicCondition.ndjson.gz` — ICD-9/10 codes A40/A41/R65.2 and 995.91/995.92/785.52
- **ICU encounters** from `MimicEncounterICU.ndjson.gz` — links observations to ICU stays via encounter references
- **Comorbidities** from `MimicCondition.ndjson.gz` — ICD-10 prefix matching (I10->hypertension, E11->diabetes, N18->CKD, J44->COPD, I50->heart failure)

Output: Same DataFrame schema as `mimic_loader.py` — patient_id, timestamp, age_years, vitals columns, lab columns, comorbidity flags, sepsis_label.

FHIR-specific handling:
- Patient age: computed from `birthDate` relative to encounter date (MIMIC uses shifted dates)
- Vital codes: extracted from `code.coding[].code` where system matches MIMIC chartevents CodeSystem
- Values: extracted from `valueQuantity.value` with unit conversion (Fahrenheit->Celsius for itemid 223761)
- GCS: sum of eye (223901) + verbal (223900) + motor (220739) components matched by encounter + time window
- Patient linkage: FHIR `subject.reference` -> Patient UUID -> MIMIC subject_id from identifier

### Modified: `src/sepsis_vitals/ml/mimic_loader.py`

The existing loader expects `hosp/patients.csv.gz`, `icu/chartevents.csv.gz`, etc. — the MIMIC-IV Demo 2.2 download provides exactly this structure.

Changes:
- Update `_validate_paths()` to handle the demo directory layout
- Add `from_demo()` classmethod that points to `physionet.org/files/mimic-iv-demo/2.2/`
- Both the FHIR and CSV loaders read the same underlying 100-patient cohort

### New: `src/sepsis_vitals/ml/data_unifier.py`

Merges outputs from both loaders into a single deduplicated DataFrame:
- Patient ID normalization: FHIR UUIDs mapped to MIMIC subject_id integers via identifier field
- Timestamp alignment across sources
- Feature column alignment to match the 62-feature schema in `model_metadata.json`
- Deduplication by (patient_id, timestamp, vital_name)

---

## Layer 2: ML Model & Stacking Ensemble

### Goal
Retrain on real MIMIC-IV data and build a stacking ensemble with clinical sensitivity weighting.

### Modified: `src/sepsis_vitals/ml/trainer.py`

Add ensemble training pipeline:

- **Base models (4):** GradientBoosting, XGBoost, LightGBM, RandomForest — all trained with 5-fold stratified cross-validation on MIMIC-IV data
- **Out-of-fold predictions:** Each base model generates cross-validated predictions on the training set (no leaking) to train the stacker
- **Stacking meta-learner:** Logistic regression trained on the 4-column matrix of out-of-fold predictions
- **Clinical weighting:** After stacker training, compute sensitivity-at-95%-specificity for each base model. Models with higher clinical sensitivity get additional weight in the final blend via a weighted average that combines the stacker output with the clinical-weighted average
- **Calibration:** Platt scaling (sigmoid) on the final ensemble output using the validation set

### New: `src/sepsis_vitals/ml/ensemble.py`

`SepsisEnsemble` class:

```
class SepsisEnsemble:
    base_models: list[BaseModel]      # 4 trained models
    stacker: LogisticRegression        # meta-learner
    calibrator: CalibratedClassifierCV # Platt scaling
    clinical_weights: np.ndarray       # sensitivity-based weights
    imputation_medians: dict           # from training data
    feature_names: list[str]           # 62 features

    def predict(features: np.ndarray) -> float:
        # 1. Run all base models -> 4 probabilities
        # 2. Feed to stacker -> raw ensemble probability
        # 3. Blend with clinical-weighted average (configurable, default 0.6 stacker + 0.4 clinical)
        # 4. Calibrate -> final probability

    def predict_with_uncertainty(features) -> (prob, ci_lower, ci_upper):
        # CI from inter-model disagreement
        # Wider CI when models disagree = more honest uncertainty

    def explain(features) -> list[FeatureImportance]:
        # Average feature importance across base models
        # Weighted by ensemble contribution
```

Serialized as single `sepsis_ensemble.joblib` containing all components.

### Modified: `src/sepsis_vitals/ml/predictor.py`

Update `SepsisPredictor` to support both model types:
- Auto-detect from metadata: `model_type: "ensemble"` vs `"single"`
- Ensemble path: `SepsisEnsemble.predict()` + `predict_with_uncertainty()`
- Confidence intervals from inter-model disagreement (replaces staged_predict_proba heuristic)
- Risk classification, recommendations, persistence unchanged

### Modified: `src/sepsis_vitals/train.py` / `retrain.py`

New CLI options:
```bash
python -m sepsis_vitals.train --data-source mimic-demo --ensemble --output models
python -m sepsis_vitals.train --data-source mimic-fhir --ensemble --output models
python -m sepsis_vitals.train --data-source synthetic --patients 20000  # existing behavior
```

### Output Artifacts

- `models/sepsis_ensemble.joblib` — all base models + stacker + calibrator
- `models/model_metadata.json` — ensemble metrics, per-model AUROC, stacker weights, clinical weights
- `models/imputation_medians.json` — computed from MIMIC-IV training split
- `models/evaluation_report.json` — comprehensive metrics

### Validation Metrics

- Per-model: AUROC, AUPRC, sensitivity@95%spec, Brier score
- Ensemble: same metrics + lift over best individual model
- Clinical operating characteristics: sensitivity/specificity/PPV/NPV at 0.5 threshold and clinically-chosen threshold
- Fairness: AUROC stratified by age bracket and sex

---

## Layer 3: Autonomous Prediction Engine

### Goal
Continuous monitoring loop that runs predictions automatically, tracks deterioration, and escalates alerts.

### New: `src/sepsis_vitals/ml/monitor.py`

Core monitoring engine with three components:

**`PatientRegistry`**
- In-memory registry of actively monitored patients
- Per-patient state: current vitals, vitals history (rolling window), last prediction time, risk trajectory (list of recent probabilities), alert state, registration time
- Backed by SQLite (`state_store.py`) for persistence across restarts
- Thread-safe access via asyncio locks

**`MonitorLoop`**
- Async background task started in FastAPI lifespan
- Prediction cycle every 60 seconds (configurable via `MONITOR_INTERVAL_SECONDS`)
- Per-patient logic:
  - Skip if no new vitals since last prediction
  - Build feature vector from current vitals + history
  - Run ensemble prediction
  - Compute deterioration score: current risk vs trailing 3-prediction moving average
  - If risk delta > 0.1 over 3 readings, escalate alert priority
  - Fire WebSocket alert on risk level transitions (low->moderate, moderate->high, high->critical)
  - Track metrics: time-to-alert, alert-to-acknowledgment, prediction count

**`VitalsIngester`**
- Unified intake for all three prediction modes:
  - `ingest_single(patient_id, vitals, demographics?, comorbidities?)` — from UI form
  - `ingest_batch(records: list[VitalsRecord])` — from batch endpoint
  - `ingest_fhir(bundle: FHIRObservationBundle)` — from FHIR webhook
- All three normalize to internal format: `{patient_id, timestamp, vitals: dict, labs: dict}`
- Auto-registers new patients in the registry
- Triggers immediate prediction for single/batch modes (don't wait for monitor cycle)

### Modified: `src/sepsis_vitals/api.py`

New endpoints:
- `POST /monitor/register` — Register patient for active monitoring (accepts patient_id + optional initial vitals)
- `DELETE /monitor/{patient_id}` — Remove patient from monitoring
- `GET /monitor/status` — List all monitored patients with current risk level, last prediction time, trend direction
- `POST /predict` — Updated to also feed vitals into monitoring registry (smart rerun)
- `POST /predict/batch` — Updated to register all patients for ongoing monitoring

Lifespan handler starts `MonitorLoop` as background task on server boot.

### Modified: `src/sepsis_vitals/realtime/websocket.py`

Enhanced alert messages:
- `alert_type`: `"new_risk"` | `"escalation"` | `"deterioration"` | `"recovery"`
- `previous_risk_level` and `risk_delta` for context
- `deterioration_rate`: risk increase per hour
- `patient_update` message type for vitals/risk refreshes (separate from alerts)

### Modified: `src/sepsis_vitals/fhir/router.py`

Wire existing FHIR endpoint scaffolding to the ingester:
- Parse incoming FHIR Observation resources -> extract vitals/labs via LOINC/MIMIC code mapping
- Feed into `VitalsIngester.ingest_fhir()`
- Return FHIR OperationOutcome response per spec

---

## Layer 4: Simulator

### Goal
Replay real MIMIC-IV cases and generate synthetic wards for demos. Both feed through the same prediction pipeline as production data.

### New: `src/sepsis_vitals/ml/simulator.py`

Two simulation modes:

**`CaseReplay`**
- Loads a real MIMIC-IV patient's ICU stay from downloaded data (via `case_library.py`)
- Replays vitals chronologically at configurable speed (default: 24h -> 2 minutes)
- Each observation fed into `VitalsIngester.ingest_single()` at the appropriate interval
- Frontend sees real vitals arriving, risk scores changing, alerts firing
- Metadata for UI: patient subject_id, source ("Beth Israel Deaconess Medical Center"), ICU stay info, total duration, current position

**`WardSimulator`**
- Generates configurable ward using `synthetic_data.py` generator
- Default: 8-12 patients. Mix: ~2 active sepsis (various stages), ~3 sick-but-not-septic confounders, ~5 stable
- Runs on async loop, generating new observations every few seconds (simulating accelerated real-time)
- One patient scripted to deteriorate from low->critical over the demo period (guaranteed escalation demo)
- All observations flow through `VitalsIngester` -> `MonitorLoop` -> `SepsisEnsemble` -> WebSocket alerts

Both simulators use the real prediction pipeline. No separate demo path.

### New: `src/sepsis_vitals/ml/case_library.py`

Indexes downloaded MIMIC-IV demo data for quick case lookup:
- Builds lightweight SQLite index of all 100 patients on first access
- Fields: subject_id, age, sex, sepsis_diagnosis (bool), icu_los_hours, n_chartevent_observations
- `get_sepsis_cases() -> list[CaseSummary]` — patients with sepsis ICD codes
- `get_case(subject_id) -> PatientTimeline` — full vitals timeline for replay
- `get_random_case(sepsis: bool = True) -> PatientTimeline` — random case matching criteria

### Modified: `src/sepsis_vitals/api.py`

New endpoints (gated behind `ENABLE_SIMULATOR=true`):
- `POST /simulator/replay` — Start replaying a MIMIC-IV patient. Accepts `{subject_id: int | "random", speed: float, sepsis_only: bool}`. Returns session_id.
- `POST /simulator/ward` — Start ward simulation. Accepts `{n_patients: int, speed: float, sepsis_count: int}`. Returns session_id.
- `DELETE /simulator/{session_id}` — Stop a running simulation
- `GET /simulator/sessions` — List active simulations with progress
- `GET /simulator/cases` — List available MIMIC-IV cases for replay selection

---

## Layer 5: Frontend UI

### Goal
Three new/overhauled pages connected to the autonomous prediction engine.

### New: `frontend/src/pages/Monitor.tsx`

Real-time ward monitoring — the central station display:

- **Patient grid**: Cards for all actively monitored patients. Each card shows:
  - Patient ID and demographics
  - Current risk badge (RiskBadge component) with level
  - Sparkline of last 12 risk scores (Recharts LineChart, minimal, no axes)
  - Latest vitals snapshot: HR, Temp, SBP, SpO2 with abnormal values highlighted
  - Time since last observation ("2m ago", "15m ago")
  - Trend arrow: improving (green down), stable (gray horizontal), worsening (red up)
- **Auto-sort by risk**: Critical patients float to top with pulsing red border. Low-risk patients compact at bottom.
- **Live WebSocket updates**: Risk scores, vitals, alert states update without refresh. New alerts trigger existing audio notification.
- **Simulation controls** (visible when simulator active): Control bar at top with play/pause, speed slider (0.5x-10x), session info ("Replaying Patient 10016742 — 4h into 18h ICU stay"), stop button.
- **Click-through**: Patient card navigates to `/patients/:id`

### Overhauled: `frontend/src/pages/PatientDetail.tsx`

Enhanced from basic charts to comprehensive clinical view:

- **Risk timeline**: Full risk probability over time as area chart with confidence interval band (shaded). Horizontal threshold lines at 0.25/0.50/0.75 labeled moderate/high/critical.
- **Clinical scores panel**: qSOFA, NEWS2, SIRS, Shock Index displayed as colored number cards with label and interpretation text. Updated per observation.
- **SHAP waterfall chart**: Per-prediction feature importance as horizontal waterfall. Red bars push risk up, green bars push risk down. Shows top 10 features for the most recent prediction.
- **Vitals history**: Multi-line chart with toggleable vitals (HR, Temp, RR, SBP, SpO2). Normal range bands as subtle colored fills. Click to toggle series visibility.
- **Alert history timeline**: Chronological list of all alerts for this patient. Shows acknowledgment status, clinical notes, response time. Uses existing AlertFeed component filtered by patient_id.
- **Deterioration indicator**: Large trend arrow with rate text ("Worsening: +12% risk/hour")

### Overhauled: `frontend/src/pages/Predict.tsx`

Full prediction workflow with all model inputs exposed:

- **Demographics section**: Age (number input), Sex (select: M/F). Pre-filled if patient exists in system.
- **Comorbidities section**: 5 checkboxes — Hypertension, Diabetes, CKD, COPD, Heart Failure. Currently default to 0 silently; now explicitly shown.
- **Vitals section**: Existing 9-vital form (temperature through procalcitonin), unchanged
- **Clinical scores display**: After prediction, show computed qSOFA, NEWS2, SIRS, Shock Index in a row of cards alongside the ML probability. Clinicians trust scoring systems they recognize.
- **Historical comparison**: If patient_id has prior predictions, show mini risk trajectory line. Text: "Risk increased from 32% -> 48% since last assessment 4 hours ago."
- **Auto-monitor toggle**: "Add to continuous monitoring" checkbox. When checked, patient registered in monitor loop after prediction.
- **Results layout**: Risk probability (large number), confidence interval, risk badge, recommendation, SHAP factors chart — same as current but with clinical scores added.

### New: `frontend/src/components/SimulatorPanel.tsx`

Floating control panel for simulator (bottom-right, collapsible):
- **Replay tab**: Dropdown of available MIMIC-IV cases (labeled with sepsis/no-sepsis, age, sex, ICU LOS). "Start Replay" button.
- **Ward sim tab**: Patient count slider (4-20), speed multiplier (1x-10x), sepsis count. "Start Ward" button.
- **Active sessions**: List with play/pause/stop per session, progress bar for replays.
- **Visibility**: Only rendered when backend reports `simulator_enabled: true` in health check.

### Modified: `frontend/src/stores/useStore.ts`

New state:
- `monitoredPatients: Record<string, MonitoredPatient>` — patient_id -> current state (risk, vitals, trend, last_update)
- `simulatorSessions: SimSession[]` — active simulation sessions
- `updatePatientRisk(id, prediction)` — update from WebSocket
- `setMonitoredPatients(patients)` — bulk update from REST polling
- `addSimSession(session)` / `removeSimSession(id)` — simulator state

### Modified: `frontend/src/hooks/useWebSocket.ts`

Handle new WebSocket message types:
- `patient_update` — vitals/risk update for monitored patient -> `updatePatientRisk()`
- `deterioration_alert` — escalation alert with previous_risk_level, risk_delta
- `simulator_event` — simulation progress (position in timeline, completion)

### Modified: `frontend/src/App.tsx`

- New route: `/monitor` -> `Monitor` page (lazy loaded)
- Nav entry: Activity icon, "Monitor" label (added to Sidebar + BottomNav)

### Modified: `frontend/src/types/index.ts`

New types:
- `MonitoredPatient`: { id, risk_level, risk_probability, vitals, trend_direction, last_update, alert_count }
- `SimSession`: { id, type: "replay" | "ward", status, progress?, patient_count, started_at }
- `ClinicalScores`: { qsofa, news2, sirs, shock_index }
- `DeteriorationAlert`: extends Alert with previous_risk_level, risk_delta, deterioration_rate

---

## Implementation Order

1. **Layer 1: Data Pipeline** — fhir_loader.py, mimic_loader.py fixes, data_unifier.py
2. **Layer 2: ML Model** — ensemble.py, trainer.py ensemble mode, retrain on MIMIC-IV
3. **Layer 3: Prediction Engine** — monitor.py (registry, loop, ingester), API endpoints, WebSocket enhancements
4. **Layer 4: Simulator** — case_library.py, simulator.py (replay + ward), simulator API endpoints
5. **Layer 5: UI** — Monitor page, PatientDetail overhaul, Predict overhaul, SimulatorPanel, store/WebSocket updates

Each layer is independently testable and adds demonstrable value.

---

## Technical Constraints

- **100-patient dataset**: MIMIC-IV Demo has only 100 patients. Ensemble must be robust to small data — heavy cross-validation, regularization, no deep learning.
- **HIPAA note**: MIMIC-IV Demo is de-identified Open Access data. No PHI concerns for the demo dataset. Production deployments with real patient data require full HIPAA compliance.
- **Model disclaimer**: All predictions carry "Research Use Only — Not FDA Cleared" labeling in the UI.
- **Existing architecture**: All new code integrates with existing FastAPI + React + Zustand + Recharts stack. No new frameworks.
- **Demo mode**: Frontend `isDemo` flag continues to work for GitHub Pages deployment. New features gracefully degrade — Monitor page shows simulated data, Predict page works with demo predictions.
