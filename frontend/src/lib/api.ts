const BASE = import.meta.env.VITE_API_URL ?? ''

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const token = localStorage.getItem('sv_token')
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  }
  const res = await fetch(`${BASE}${path}`, { ...options, headers })
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(body.detail ?? `HTTP ${res.status}`)
  }
  return res.json()
}

/** True when running on GitHub Pages (no backend available). */
export const isDemo = window.location.hostname.includes('github.io')

export const api = {
  health: () => request<{ status: string; version: string }>('/health'),

  score: (vitals: Record<string, number>) =>
    request('/score', { method: 'POST', body: JSON.stringify(vitals) }),

  predict: (body: { vitals: Record<string, number>; patient_id: string; age_years?: number }) =>
    request('/predict', { method: 'POST', body: JSON.stringify(body) }),

  predictBatch: (patients: Array<{ vitals: Record<string, number>; patient_id: string }>) =>
    request('/predict/batch', { method: 'POST', body: JSON.stringify({ patients }) }),

  copilot: (body: { vitals: Record<string, number>; patient_id: string; question?: string }) =>
    request('/copilot', { method: 'POST', body: JSON.stringify(body) }),

  patientTrend: (patientId: string) =>
    request(`/patient/${patientId}/trend`),

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
}
