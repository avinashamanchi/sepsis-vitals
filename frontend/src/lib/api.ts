// Default '/api': the Vite dev proxy and the nginx container both strip this
// prefix before forwarding to the FastAPI backend, which has no /api routes.
// `||` (not `??`) so a blank VITE_API_URL in .env also falls back.
const BASE = import.meta.env.VITE_API_URL || '/api'

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

/** An HTTP error from the API, with a message a person can read. */
export class ApiError extends Error {
  readonly status: number
  readonly detail: unknown

  constructor(status: number, detail: unknown) {
    super(describeDetail(status, detail))
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

/**
 * FastAPI returns `detail` as a string, an object (for example the 503 when no
 * usable model is installed) or a list of validation errors. Never show
 * "[object Object]".
 */
export function describeDetail(status: number, detail: unknown): string {
  if (typeof detail === 'string' && detail) return detail
  if (Array.isArray(detail)) {
    const messages = detail
      .map((d) => (d && typeof d === 'object' && 'msg' in d ? String((d as { msg: unknown }).msg) : ''))
      .filter(Boolean)
    if (messages.length) return `Invalid input: ${messages.join('; ')}`
  }
  if (detail && typeof detail === 'object') {
    const d = detail as { message?: unknown; model_state?: unknown }
    if (typeof d.message === 'string') {
      return typeof d.model_state === 'string' ? `${d.message} (model state: ${d.model_state})` : d.message
    }
  }
  return `Request failed (HTTP ${status})`
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
    throw new ApiError(res.status, body?.detail)
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

export type LoginResult =
  | { kind: 'session'; access_token: string; refresh_token?: string; user?: { email: string; role: string } }
  | { kind: 'mfa_required' }
  | { kind: 'mfa_enrollment_required'; enrollmentToken: string }

/** For the MFA enrollment endpoints, which take a short-lived enrollment token. */
async function requestWithToken<T>(path: string, token: string, options: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...options,
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
  })
  const body = await res.json().catch(() => ({}))
  if (!res.ok) throw new ApiError(res.status, body?.detail)
  return body as T
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

  /**
   * Sign in. MFA outcomes are returned, not thrown: the backend answers 401
   * "mfa_required" when a code is needed and 403 "mfa_enrollment_required"
   * (with an enrollment-only token) when the user's role requires MFA.
   */
  login: async (email: string, password: string, otp?: string): Promise<LoginResult> => {
    const res = await fetch(`${BASE}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(otp ? { email, password, otp } : { email, password }),
    })
    const body = await res.json().catch(() => ({}))
    if (res.ok) return { kind: 'session', ...body }
    if (res.status === 401 && body.detail === 'mfa_required') return { kind: 'mfa_required' }
    if (res.status === 403 && body.detail === 'mfa_enrollment_required') {
      return { kind: 'mfa_enrollment_required', enrollmentToken: body.enrollment_token }
    }
    throw new Error(typeof body.detail === 'string' ? body.detail : `HTTP ${res.status}`)
  },

  mfaEnroll: (enrollmentToken: string) =>
    requestWithToken<{ secret: string; otpauth_uri: string }>(
      '/auth/mfa/enroll', enrollmentToken, { method: 'POST' },
    ),

  mfaConfirm: (enrollmentToken: string, code: string) =>
    requestWithToken<{ recovery_codes: string[] }>(
      '/auth/mfa/confirm', enrollmentToken, { method: 'POST', body: JSON.stringify({ code }) },
    ),

  weeklyTrends: (days = 7) =>
    request<Array<{ date: string | null; predictions: number; alerts: number }>>(
      `/patients/dashboard/weekly-trends?days=${days}`,
    ),

  riskDistribution: (hoursBack = 24) =>
    request<Array<{ risk_level: string; count: number; percentage: number }>>(
      `/patients/dashboard/risk-distribution?hours_back=${hoursBack}`,
    ),

  confirmPasswordReset: (token: string, newPassword: string) =>
    request<{ detail: string }>('/auth/password-reset/confirm', {
      method: 'POST',
      body: JSON.stringify({ token, new_password: newPassword }),
    }),

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

  // Mirrors PatientSummaryOut. latest_* are null until a patient is observed:
  // render that as "not yet observed", never as low risk or 0.
  getPatients: (siteId?: string) =>
    request<Array<{
      id: string
      external_id: string
      site_id: string
      age_years: number | null
      sex: string | null
      latest_vitals: Record<string, number> | null
      latest_risk_level: string | null
      latest_recorded_at: string | null
    }>>(`/patients${siteId ? `?site_id=${encodeURIComponent(siteId)}` : ''}`),

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

  // The backend scopes stats to the signed-in user's site; only
  // administrators may pass an explicit site.
  dashboardStats: (siteId?: string) =>
    request<{
      patient_count: number
      active_alerts: number
      recent_predictions: number
    }>(`/patients/dashboard/stats${siteId ? `?site_id=${encodeURIComponent(siteId)}` : ''}`),

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
