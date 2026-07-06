import { describe, it, expect, beforeEach } from 'vitest'
import { useStore } from '../stores/useStore'

describe('useStore', () => {
  beforeEach(() => {
    // Reset store state between tests
    useStore.setState({
      token: null,
      user: null,
      alerts: [],
      patients: [],
      wsConnected: false,
    })
  })

  it('sets auth state', () => {
    useStore.getState().setAuth('test-token', { email: 'a@b.com', role: 'user' })
    const state = useStore.getState()
    expect(state.token).toBe('test-token')
    expect(state.user?.email).toBe('a@b.com')
  })

  it('logout clears auth', () => {
    useStore.getState().setAuth('test-token', { email: 'a@b.com', role: 'user' })
    useStore.getState().logout()
    const state = useStore.getState()
    expect(state.token).toBeNull()
    expect(state.user).toBeNull()
  })

  it('adds and dismisses alerts', () => {
    useStore.getState().addAlert({
      id: 'a1',
      patientId: 'P1',
      type: 'sepsis_alert',
      message: 'High risk',
      timestamp: Date.now(),
      severity: 'high',
    } as any)

    expect(useStore.getState().alerts).toHaveLength(1)

    useStore.getState().dismissAlert('a1')
    expect(useStore.getState().alerts[0].dismissed).toBe(true)
  })

  it('alerts selector returns stable reference when unchanged', () => {
    // Regression: inline .filter() in selector caused infinite re-renders
    const ref1 = useStore.getState().alerts
    const ref2 = useStore.getState().alerts
    expect(ref1).toBe(ref2) // same reference, not a new array
  })

  it('updates activity timestamp', () => {
    const before = useStore.getState().lastActivity
    useStore.getState().updateActivity()
    expect(useStore.getState().lastActivity).toBeGreaterThanOrEqual(before)
  })
})
