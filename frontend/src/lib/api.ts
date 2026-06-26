const BASE = import.meta.env.VITE_API_URL ?? ''

/** Callback set by the app to handle forced logouts (401). */
let onUnauthorized: (() => void) | null = null

/** Register a callback invoked on 401 responses. */
export function setOnUnauthorized(cb: () => void) {
  onUnauthorized = cb
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const token = localStorage.getItem('sv_token')
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  }
  const res = await fetch(`${BASE}${path}`, { ...options, headers })
  if (!res.ok) {
    if (res.status === 401 && onUnauthorized) {
      onUnauthorized()
    }
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(body.detail ?? `HTTP ${res.status}`)
  }
  return res.json()
}

/** True when running on GitHub Pages (no backend available). */
export const isDemo = window.location.hostname.includes('github.io')

export const api = {
  health: () => request<{ status: string; version: string }>('/health'),

  login: (email: string, password: string) =>
    request<{ access_token: string; user: { email: string; role: string } }>(
      '/auth/login',
      { method: 'POST', body: JSON.stringify({ email, password }) },
    ),

  register: (email: string, password: string) =>
    request<{ access_token: string; user: { email: string; role: string } }>(
      '/auth/register',
      { method: 'POST', body: JSON.stringify({ email, password }) },
    ),

  score: (vitals: Record<string, number>) =>
    request('/score', { method: 'POST', body: JSON.stringify(vitals) }),

  predict: (body: { vitals: Record<string, number>; patient_id: string; age_years?: number }) =>
    request('/predict', { method: 'POST', body: JSON.stringify(body) }),

  predictBatch: (patients: Array<{ vitals: Record<string, number>; patient_id: string }>) =>
    request('/predict/batch', { method: 'POST', body: JSON.stringify({ patients }) }),

  copilot: (body: { vitals: Record<string, number>; patient_id: string; question?: string }) =>
    request('/copilot', { method: 'POST', body: JSON.stringify(body) }),

  getPatients: (siteId?: string) =>
    request<Array<{
      id: string
      name?: string
      bed?: string
      vitals: Record<string, number>
      riskLevel: string
      riskProbability: number
      lastUpdated: string
    }>>(`/patients/${siteId ? `?site_id=${siteId}` : ''}`),

  patientTrend: (patientId: string) =>
    request<{
      patient_id: string
      trend: Array<{
        timestamp: string
        risk_probability: number
        vitals: Record<string, number>
      }>
    }>(`/patient/${patientId}/trend`),

  modelInfo: () =>
    request<{
      model_name: string
      version: string
      is_calibrated: boolean
      feature_count: number
      metrics: Record<string, number>
      feature_importance: Record<string, number>
    }>('/model/info'),

  dashboardStats: (siteId: string = 'default') =>
    request<{
      patient_count: number
      active_alerts: number
      predictions_today: number
      avg_response_min: number | null
    }>(`/patients/dashboard/stats?site_id=${siteId}`),

  systemHealth: () =>
    request<{
      status: string
      version: string
      model_loaded: boolean
      database: string
      redis: string
      websocket_connections: number
      uptime_seconds: number
    }>('/health'),

  // Monitor endpoints
  monitorRegister: (patientId: string, demographics?: Record<string, unknown>, comorbidities?: Record<string, number>) =>
    request<{ status: string; patient_id: string }>(
      '/monitor/register',
      { method: 'POST', body: JSON.stringify({ patient_id: patientId, demographics, comorbidities }) },
    ),

  monitorUnregister: (patientId: string) =>
    request<{ status: string; patient_id: string }>(
      `/monitor/${patientId}`,
      { method: 'DELETE' },
    ),

  monitorStatus: () =>
    request<{
      patients: Array<{
        patient_id: string
        demographics: Record<string, string | number>
        vitals: Record<string, number>
        risk_probability: number
        risk_level: string
        trend_direction: string
        last_prediction_time: number
        last_vitals_time: number
        registered_at: number
        alert_state: string
        deterioration_rate: number
        window_hours: number
      }>
      count: number
    }>('/monitor/status'),

  // Simulator endpoints
  simulatorStartWard: (opts: { n_patients?: number; speed?: number; sepsis_count?: number; seed?: number }) =>
    request<{ session_id: string; status: string }>(
      '/simulator/ward',
      { method: 'POST', body: JSON.stringify(opts) },
    ),

  simulatorStartReplay: (opts: { subject_id?: number | string; speed?: number; sepsis_only?: boolean }) =>
    request<{ session_id: string; subject_id: number; status: string }>(
      '/simulator/replay',
      { method: 'POST', body: JSON.stringify(opts) },
    ),

  simulatorStop: (sessionId: string) =>
    request<{ session_id: string; status: string }>(
      `/simulator/${sessionId}`,
      { method: 'DELETE' },
    ),

  simulatorSessions: () =>
    request<{ sessions: Array<{ session_id: string; type: string; status: string; patient_count?: number; started_at: number }> }>(
      '/simulator/sessions',
    ),

  simulatorCases: () =>
    request<{
      cases: Array<{
        subject_id: number
        hadm_id: number
        stay_id: number
        age_years: number
        sex: string
        sepsis_label: number
        icu_los_hours: number
        n_observations: number
      }>
      count: number
    }>('/simulator/cases'),
}
