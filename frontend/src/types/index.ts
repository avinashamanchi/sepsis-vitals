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

export interface CopilotResponse {
  analysis: string
  risk_level: RiskLevel
  key_concerns: string[]
  suggested_actions: string[]
  disclaimer: string
}
