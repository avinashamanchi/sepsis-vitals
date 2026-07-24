const BASE = import.meta.env.VITE_API_URL ?? ''

/** Session-scoped tokens reduce exposure if a shared clinical workstation is left behind. */
function safeGetItem(key: string): string | null {
  try {
    return sessionStorage.getItem(key)
  } catch {
    return null
  }
}

/** Callback set by the app to handle forced logouts (401). */
let onUnauthorized: (() => void) | null = null

/** Register a callback invoked on 401 responses. */
export function setOnUnauthorized(cb: () => void) {
  onUnauthorized = cb
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const token = safeGetItem('sv_token')
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  }

  let res: Response
  try {
    res = await fetch(`${BASE}${path}`, { ...options, headers })
  } catch (err) {
    throw new Error(
      err instanceof Error ? err.message : 'Network error — check your connection',
      { cause: err },
    )
  }

  if (!res.ok) {
    if (res.status === 401 && onUnauthorized) {
      onUnauthorized()
    }
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(body.detail ?? `HTTP ${res.status}`)
  }

  // Handle empty responses (204 No Content, etc.)
  const text = await res.text()
  if (!text) return {} as T
  try {
    return JSON.parse(text) as T
  } catch {
    throw new Error('Invalid JSON response from server')
  }
}

/** Explicit public-demo mode; GitHub Pages remains the default demo host. */
export const isDemo =
  import.meta.env.VITE_DEMO_MODE === 'true' ||
  window.location.hostname.includes('github.io')

function simulateDemoPrediction(body: {
  vitals: Record<string, number>
  patient_id: string
}) {
  const vitals = body.vitals
  const signals = [
    ['resp_rate', vitals.resp_rate != null && vitals.resp_rate >= 22, vitals.resp_rate ?? 0],
    ['heart_rate', vitals.heart_rate != null && vitals.heart_rate > 100, vitals.heart_rate ?? 0],
    ['sbp', vitals.sbp != null && vitals.sbp <= 100, vitals.sbp ?? 0],
    ['temperature', vitals.temperature != null && (vitals.temperature < 36 || vitals.temperature > 38), vitals.temperature ?? 0],
    ['spo2', vitals.spo2 != null && vitals.spo2 < 94, vitals.spo2 ?? 0],
    ['lactate', vitals.lactate != null && vitals.lactate >= 2, vitals.lactate ?? 0],
  ] as const
  const active = signals.filter(([, present]) => present)
  const probability = Math.min(0.92, 0.08 + active.length * 0.14)
  const riskLevel =
    probability >= 0.7 ? 'critical' :
      probability >= 0.5 ? 'high' :
        probability >= 0.25 ? 'moderate' : 'low'
  const qsofa =
    Number((vitals.sbp ?? 999) <= 100) +
    Number((vitals.resp_rate ?? 0) >= 22) +
    Number((vitals.gcs ?? 15) < 15)
  const sirs =
    Number((vitals.temperature ?? 37) < 36 || (vitals.temperature ?? 37) > 38) +
    Number((vitals.heart_rate ?? 0) > 90) +
    Number((vitals.resp_rate ?? 0) > 20) +
    Number(vitals.wbc != null && (vitals.wbc < 4 || vitals.wbc > 12))

  return Promise.resolve({
    patient_id: body.patient_id,
    timestamp: new Date().toISOString(),
    risk_probability: probability,
    risk_level: riskLevel,
    confidence_interval: {
      lower: Math.max(0, probability - 0.12),
      upper: Math.min(1, probability + 0.12),
    },
    alert: probability >= 0.5,
    clinical_scores: {
      qsofa,
      sirs_count: sirs,
      news2_style: Math.min(12, active.length * 2),
      shock_index: vitals.heart_rate && vitals.sbp
        ? vitals.heart_rate / vitals.sbp
        : null,
    },
    top_risk_factors: active.map(([feature], index) => ({
      feature,
      importance: Number((0.32 - index * 0.04).toFixed(2)),
    })),
    recommendation:
      'Synthetic UI simulation only. Record the result for workflow evaluation; do not use it to guide diagnosis or treatment.',
    model: { name: 'Synthetic interface simulator', version: 'demo' },
    research_only: true,
    validation_status: 'Synthetic demonstration; not a model inference',
    intended_use: 'Interface evaluation only',
  })
}

export const api = {
  health: () => request<{ status: string; version: string }>('/health'),

  login: (email: string, password: string) =>
    request<{
      access_token: string
      refresh_token: string
      user?: { email: string; role: string }
    }>(
      '/auth/login',
      { method: 'POST', body: JSON.stringify({ email, password }) },
    ),

  requestPasswordReset: (email: string) =>
    request<{ detail: string }>('/auth/password-reset/request', {
      method: 'POST',
      body: JSON.stringify({ email }),
    }),

  logout: () =>
    request<{ detail: string }>('/auth/logout', {
      method: 'POST',
      body: JSON.stringify({}),
    }),

  score: (vitals: Record<string, number>) =>
    request('/score', { method: 'POST', body: JSON.stringify(vitals) }),

  predict: (body: { vitals: Record<string, number>; patient_id: string; age_years?: number }) =>
    isDemo
      ? simulateDemoPrediction(body)
      : request('/predict', { method: 'POST', body: JSON.stringify(body) }),

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

  ping: () => request<{ status: string }>('/auth/ping', { method: 'POST' }),

  // Alert lifecycle
  alertAcknowledge: (alertId: string) =>
    request<Record<string, unknown>>(`/alerts/ack/${alertId}`, { method: 'POST', body: JSON.stringify({}) }),

  alertResolve: (alertId: string, reason?: string) =>
    request<Record<string, unknown>>(`/alerts/resolve/${alertId}`, { method: 'POST', body: JSON.stringify({ reason }) }),

  alertSnooze: (alertId: string, minutes: number = 15) =>
    request<Record<string, unknown>>(`/alerts/snooze/${alertId}`, { method: 'POST', body: JSON.stringify({ minutes }) }),
}
