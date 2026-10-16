# Inference contract

How a risk result is requested, what it contains, how it is recorded, and
what happens when something fails. It covers `POST /predict`,
`POST /predict/timeseries`, `POST /predict/batch` and the continuous monitor.

**Not approved for clinical use.** Every result is decision support for
research evaluation, from a model developed on synthetic data. It is not a
diagnosis. `clinical_use` is always `not-permitted`.

## 1. Requests

### `POST /predict`: one observation

```json
{"patient_id": "synthetic-001", "age_years": 70,
 "vitals": {"heart_rate": 118, "resp_rate": 24, "sbp": 96, "temperature": 38.6},
 "comorbidities": {"has_diabetes": 1}}
```

History is the patient's recorded vitals: the last 4 readings, and only for
registered patients.

### `POST /predict/timeseries`: a validated series for one patient

```json
{"patient_id": "synthetic-001", "age_years": 70,
 "units": {"temperature": "Cel", "heart_rate": "/min", "sbp": "mm[Hg]"},
 "observations": [
   {"timestamp": "2026-10-15T08:00:00Z", "heart_rate": 92, "resp_rate": 18, "sbp": 124, "temperature": 37.4},
   {"timestamp": "2026-10-15T09:00:00Z", "heart_rate": 104, "resp_rate": 22, "sbp": 110, "temperature": 38.1},
   {"timestamp": "2026-10-15T10:00:00Z", "heart_rate": 118, "resp_rate": 24, "sbp": 96, "temperature": 38.6}
 ]}
```

The last observation is the one scored, and the earlier ones are its
history. The payload is the complete input: no recorded vitals are added.

A request is refused with 422 and the reason in any of these cases:

| Check | Rule |
|---|---|
| Schema | Unknown fields are refused, never ignored. This applies to the observation, the comorbidities and the request. |
| Units | Values must already be in the canonical unit (table below). A `units` declaration must match it; nothing is converted. |
| Range | Each value must be within the bounds in `schemas.VitalsInput`, for example temperature 25–45 °C. |
| Timestamps | Every timestamp needs a UTC offset. |
| Ordering | Observations must be in strictly ascending time order. |
| Duplicates | Two observations with the same timestamp are refused. |
| Patient | An observation's `patient_id`, if given, must equal the request's. |
| Window | The newest observation may be at most 5 minutes in the future (clock skew). The limit is 48 observations, of which only the last 4 earlier ones affect the score. |
| Content | The newest observation needs at least 3 measurements. |

| Field | Unit | Accepted declarations |
|---|---|---|
| temperature | °C | `Cel`, `°C`, `C`, `degC` |
| heart_rate, resp_rate | per minute | `/min`, `bpm`, `beats/min`, `breaths/min` |
| sbp, dbp, map | mmHg | `mm[Hg]`, `mmHg` |
| spo2 | % | `%` |
| gcs | score 3–15 | `{score}`, `score` |
| lactate | mmol/L | `mmol/L` |
| wbc | 10⁹/L | `10*9/L`, `10^9/L`, `x10^9/L` |
| procalcitonin | ng/mL | `ng/mL` |

The 5-minute skew and the 48-observation limit are engineering limits. A
maximum acceptable age for the newest observation would be a clinical
decision, so none is enforced. The age of the data is reported instead
(`observed_at`, `data_quality.history_span_minutes`).

## 2. Processing

1. **Validation**: as above.
2. **Features**: the training pipeline (`trainer.prepare_features`) over the history and the current observation.
3. **Preprocessing**: training-median imputation, then the artifact's scaler if it has one. This happens exactly once per row (tested).
4. **Model**: the verified artifact (`models/manifest.json`).
5. **Bands and alert**: the threshold policy (`ml/decision_policy.py`), combined with the rule-based level, which the model can raise but never lower.
6. **Attribution**: for each input the patient supplied, the model is re-scored with that input set to its training median. All variants go through the same single preprocessing pass.
7. **Audit record**: written and committed.
8. **Release**: only now is the result returned or broadcast.

## 3. Result

| Field | Meaning |
|---|---|
| `risk_probability` | Development-model output. **Not a probability for patients.** |
| `rule_risk_level`, `model_risk_level`, `risk_level` | The rule level, the model band, and the higher of the two. |
| `alert` | The rule-based alert, or a combined level of high/critical, or a model output above `alert_cutoff`. |
| `threshold_policy` | The id, status, bands, alert cut-off, rule and SHA-256 digest of the cut-offs in force. **Placeholder values (`placeholder-bands-v1`), not clinically derived or approved.** |
| `confidence_interval` | `null` unless the artifact supports a validated interval. The committed artifact does not. |
| `uncertainty` | `method: "none"` and a note for the committed artifact. With a calibrated split-conformal predictor, it holds a label set at a stated coverage instead, which is not an interval for the patient's risk. |
| `top_risk_factors`, `explanation` | Per-patient attribution by median substitution: `contribution` is this output minus the output with that input at its training median. It describes the model, not physiology, and is not causal. |
| `data_quality` | Warnings (fields not measured and imputed, age not supplied so 55 was used, comorbidities not supplied, no history, history ignored), observations used, history span. |
| `versions` | Model version, artifact SHA-256, feature-schema SHA-256, preprocessing (pipeline) version, threshold-policy id and digest. |
| `observed_at`, `timestamp` | When the scored observation was taken; when it was scored (UTC). |
| `request_id`, `audit_record_id` | Correlation with the request log and the audit record. |
| `decision_support_notice`, `clinical_use`, `validation_status` | Fixed disclaimers. |

## 4. Audit record (`prediction_records`)

There is one row per released result, from every path: `api`, `api-batch`,
`api-timeseries`, `monitor` and `copilot`. It is written before the result
leaves the service.

- **Input:** the canonical payload (observation, history used, age, comorbidities, observation time) in `input_vitals`, and its SHA-256 in `input_sha256`.
- **Output:** levels, scores, versions, the full threshold policy, uncertainty, explanation, data quality and the monitor's alert decision, in `output_scores`.
- **Keys:** `threshold_policy` (id), `request_id`, `source`, `created_at`.
- **Seal:** `record_sha256` covers the content. `audit_records.verify_record` returns False if the row changed, and None for rows written before migration 006.
- **Append-only:** database triggers refuse UPDATE, DELETE and TRUNCATE on PostgreSQL (migration 006) and on SQLite. ORM guards refuse them earlier. A retention policy, once approved, will need a separate privileged procedure.

## 5. Failure behaviour (fail closed)

| Failure | `/predict*` | Continuous monitor |
|---|---|---|
| No usable model | 503 with `model_state` | No ingester; readings are stored, nothing is scored |
| Model raises while scoring | 503 "No risk calculated … not a low-risk result" | `risk_unavailable` broadcast; the patient's previous score is cleared |
| Audit record cannot be written | 503 "No risk calculated … withheld" | `risk_unavailable` broadcast; tracker state restored; the previous score is cleared |
| Invalid input | 422 with the reason | — |

No path returns a default, partial or previous score. The UI shows
"No risk calculated" and never 0 % or "low" in place of a missing result.

## 6. Logs

Each request produces one `sepsis_vitals.requests` line (JSON) with:

- the request id;
- the method;
- the **route template**;
- the status;
- the latency;
- for inference, the outcome, model version and error class.

Request bodies, patient identifiers, user identifiers and raw paths are not
logged. A client-supplied `X-Request-ID` is used only if it matches
`[A-Za-z0-9-]{8,64}`.

## 7. Open clinical decisions

These are not made in code, and need clinical governance
(`docs/pilot/clinical_requirements.md`):

- thresholds and the alert rule;
- the deterioration rule;
- a prediction horizon;
- a maximum observation age;
- whether an alert may be withheld when the audit store is unavailable (the current engineering default is to withhold it);
- reference ranges.
