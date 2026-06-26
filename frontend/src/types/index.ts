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

export interface Prediction {
  patient_id: string
  timestamp: string
  risk_probability: number
  risk_level: RiskLevel
  confidence_interval: { lower: number; upper: number }
  alert: boolean
  clinical_scores: Record<string, number>
  top_risk_factors: Array<{ feature: string; importance: number }>
  recommendation: string
}

export interface ScoreResult {
  qsofa: number
  sirs_count: number
  news2_style: number
  shock_index: number | null
  uva: number
  risk_level: RiskLevel
  alert_flag: boolean
  explanations: string[]
}

export interface MonitoredPatient {
  patient_id: string
  demographics: Record<string, string | number>
  vitals: Record<string, number>
  risk_probability: number
  risk_level: RiskLevel
  trend_direction: 'improving' | 'stable' | 'worsening' | 'unknown'
  last_prediction_time: number
  last_vitals_time: number
  registered_at: number
  alert_state: 'normal' | 'elevated' | 'escalated' | 'critical'
  deterioration_rate: number
  window_hours: number
  risk_history: Array<{ timestamp: number; risk_probability: number }>
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

