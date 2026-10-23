export type RiskLevel = 'low' | 'moderate' | 'high' | 'critical'

export interface Vitals {
  temperature?: number
  heart_rate?: number
  resp_rate?: number
  sbp?: number
  dbp?: number
  spo2?: number
  gcs?: number
  map?: number
  lactate?: number
  wbc?: number
  procalcitonin?: number
}

export interface Patient {
  id: string
  name?: string
  bed?: string
  vitals: Vitals
  riskLevel: RiskLevel
  riskProbability: number
  lastUpdated: string
  alerts: Alert[]
}

export interface Alert {
  id: string
  patientId: string
  riskLevel: RiskLevel
  riskProbability: number
  message: string
  timestamp: string
  dismissed: boolean
  acknowledged?: boolean
  note?: string
}

/** Per-patient attribution: change in the model output when this input is
 * replaced by its training median. Describes the model, not physiology. */
export interface RiskFactor {
  feature: string
  label?: string
  contribution: number
  direction?: 'raised' | 'lowered' | 'none'
  current_value?: number | string[] | null
  reference_value?: number | string[] | null
}

export interface DataQualityWarning {
  code: string
  message: string
  fields: string[]
}

export interface Prediction {
  patient_id: string
  timestamp: string
  observed_at?: string | null
  /** Null in silent mode (model output recorded but withheld). */
  risk_probability: number | null
  display_mode?: 'research' | 'silent'
  model_output_withheld?: boolean
  risk_level: RiskLevel
  /** Null unless the model supports a validated interval (see uncertainty). */
  confidence_interval: { lower: number; upper: number } | null
  uncertainty?: { method: string; note?: string; coverage_target?: number; label_set?: string[] }
  alert: boolean
  clinical_scores: Record<string, number>
  top_risk_factors: RiskFactor[]
  explanation?: { method: string; note: string }
  data_quality?: {
    warnings: DataQualityWarning[]
    observations_used?: number
    history_span_minutes?: number | null
    imputed_fields?: string[]
  }
  threshold_policy?: {
    id: string
    status: string
    model_bands: Record<string, number>
    alert_cutoff: number
    alert_rule: string
    sha256: string
  }
  versions?: Record<string, string | null>
  decision_support_notice?: string
  request_id?: string
  audit_record_id?: string
  recommendation: string
  model: { name?: string; version?: string }
  research_only?: boolean
  validation_status?: string
  intended_use?: string
  /** Risk level from the rule-based scores alone. */
  rule_risk_level?: RiskLevel | 'unknown'
  /** Risk level from the model probability alone. */
  model_risk_level?: RiskLevel | 'unknown'
  /** Always "not-permitted" in this build. */
  clinical_use?: string
  provenance?: Record<string, unknown>
}

export interface ScoreResult {
  qsofa: number
  sirs_count: number
  news2_style: number
  shock_index: number | null
  /** Published UVA score (Moore et al. 2017): in-hospital mortality risk, 0-13. */
  uva: number
  uva_risk_band?: 'low' | 'medium' | 'high'
  uva_missing_inputs?: string[]
  uva_outcome?: string
  uva_reference?: string
  risk_level: RiskLevel
  alert_flag: boolean
  explanations: string[]
}

export interface MonitoredPatient {
  patient_id: string
  demographics: Record<string, string | number>
  vitals: Record<string, number>
  /** Null until a risk has been calculated and recorded, or after a failed attempt. */
  risk_probability: number | null
  risk_level: RiskLevel | 'unknown'
  /** 'scored' only when the latest attempt produced a recorded result. */
  risk_status?: 'scored' | 'not-scored' | 'rules-only' | 'unavailable'
  risk_status_reason?: string | null
  trend_direction: 'improving' | 'stable' | 'worsening' | 'unknown'
  last_prediction_time: number
  last_vitals_time: number
  registered_at: number
  alert_state: 'normal' | 'elevated' | 'escalated' | 'critical'
  deterioration_rate: number
  window_hours: number
  risk_history: Array<{ timestamp: number; risk_probability: number | null }>
}

export interface SimSession {
  session_id: string
  type: 'replay' | 'ward'
  status: 'running' | 'completed' | 'stopped'
  patient_count?: number
  started_at: number
  subject_id?: number
}

export interface ClinicalScores {
  qsofa: number
  sirs_count: number
  news2_style: number
  shock_index: number | null
}

export interface DeteriorationAlert extends Alert {
  alert_type: 'deterioration' | 'recovery' | 'escalation'
  previous_risk_level: RiskLevel
  risk_delta: number
  deterioration_rate: number
  window_hours: number
}
