import { afterEach, describe, expect, it, vi } from 'vitest'

function mockFetch(status = 200, body: unknown = {}) {
  const fn = vi.fn(async (..._args: unknown[]) =>
    new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } }),
  )
  vi.stubGlobal('fetch', fn)
  return fn
}

async function loadApi(env: Record<string, string> = {}) {
  vi.resetModules()
  for (const [key, value] of Object.entries(env)) vi.stubEnv(key, value)
  return import('../lib/api')
}

afterEach(() => {
  vi.unstubAllEnvs()
  vi.unstubAllGlobals()
  sessionStorage.clear()
})

describe('API routing', () => {
  it('defaults to /api so the dev and nginx proxies can strip the prefix', async () => {
    const fetchMock = mockFetch(200, [])
    const { api } = await loadApi({ VITE_API_URL: '' })
    await api.getPatients()
    expect(fetchMock.mock.calls[0][0]).toBe('/api/patients')
  })

  it('uses an absolute VITE_API_URL unchanged', async () => {
    const fetchMock = mockFetch(200, [])
    const { api } = await loadApi({ VITE_API_URL: 'https://api.example.org' })
    await api.getPatients()
    expect(fetchMock.mock.calls[0][0]).toBe('https://api.example.org/patients')
  })

  it('omits site_id for dashboard stats unless one is given', async () => {
    const fetchMock = mockFetch(200, { patient_count: 0, active_alerts: 0, recent_predictions: 0 })
    const { api } = await loadApi({ VITE_API_URL: '' })
    await api.dashboardStats()
    await api.dashboardStats('SITE A')
    expect(fetchMock.mock.calls[0][0]).toBe('/api/patients/dashboard/stats')
    expect(fetchMock.mock.calls[1][0]).toBe('/api/patients/dashboard/stats?site_id=SITE%20A')
  })
})

describe('authentication handling', () => {
  it('sends the session bearer token', async () => {
    const fetchMock = mockFetch(200, [])
    sessionStorage.setItem('sv_token', 'session-token')
    const { api } = await loadApi()
    await api.getPatients()
    const init = fetchMock.mock.calls[0][1] as RequestInit
    expect((init.headers as Record<string, string>).Authorization).toBe('Bearer session-token')
  })

  it('reports 401 responses to the unauthorized handler and rejects', async () => {
    mockFetch(401, { detail: 'Token has expired' })
    const { api, setOnUnauthorized } = await loadApi()
    const onUnauthorized = vi.fn()
    setOnUnauthorized(onUnauthorized)
    await expect(api.getPatients()).rejects.toThrow('Token has expired')
    expect(onUnauthorized).toHaveBeenCalledTimes(1)
  })

  it('does not treat 403 as a session expiry', async () => {
    mockFetch(403, { detail: 'Account has no site assignment' })
    const { api, setOnUnauthorized } = await loadApi()
    const onUnauthorized = vi.fn()
    setOnUnauthorized(onUnauthorized)
    await expect(api.getPatients()).rejects.toThrow('no site assignment')
    expect(onUnauthorized).not.toHaveBeenCalled()
  })

  it('posts the reset token and new password to the confirm endpoint', async () => {
    const fetchMock = mockFetch(200, { detail: 'ok' })
    const { api } = await loadApi({ VITE_API_URL: '' })
    await api.confirmPasswordReset('tok-1', 'a-long-new-password')
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit]
    expect(url).toBe('/api/auth/password-reset/confirm')
    expect(JSON.parse(init.body as string)).toEqual({ token: 'tok-1', new_password: 'a-long-new-password' })
  })
})

describe('demo mode', () => {
  it('is off on a live host unless explicitly enabled', async () => {
    expect((await loadApi({ VITE_DEMO_MODE: 'false' })).isDemo).toBe(false)
    expect((await loadApi({ VITE_DEMO_MODE: 'true' })).isDemo).toBe(true)
  })
})
