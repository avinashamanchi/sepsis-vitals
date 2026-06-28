import { create } from 'zustand'
import type { Alert, Patient, MonitoredPatient, SimSession } from '../types'

interface AppState {
  // Auth
  token: string | null
  user: { email: string; role: string } | null
  lastActivity: number
  setAuth: (token: string, user: { email: string; role: string }) => void
  logout: () => void
  updateActivity: () => void

  // Patients
  patients: Patient[]
  setPatients: (patients: Patient[]) => void
  updatePatient: (id: string, update: Partial<Patient>) => void

  // Alerts
  alerts: Alert[]
  addAlert: (alert: Alert) => void
  dismissAlert: (id: string) => void
  acknowledgeAlert: (id: string) => void
  setAlertNote: (id: string, note: string) => void
  clearAlerts: () => void

  // WebSocket
  wsConnected: boolean
  setWsConnected: (connected: boolean) => void
  wsState: 'connected' | 'reconnecting' | 'offline'
  setWsState: (state: 'connected' | 'reconnecting' | 'offline') => void

  // UI
  sidebarOpen: boolean
  setSidebarOpen: (open: boolean) => void
  showSessionWarning: boolean
  setShowSessionWarning: (show: boolean) => void

  // Monitor
  monitoredPatients: Record<string, MonitoredPatient>
  setMonitoredPatients: (patients: MonitoredPatient[]) => void
  updatePatientRisk: (patientId: string, update: Partial<MonitoredPatient>) => void
  removeMonitoredPatient: (patientId: string) => void

  // Simulator
  simulatorSessions: SimSession[]
  addSimSession: (session: SimSession) => void
  removeSimSession: (sessionId: string) => void
  updateSimSession: (sessionId: string, update: Partial<SimSession>) => void
  simulatorEnabled: boolean
  setSimulatorEnabled: (enabled: boolean) => void
}

/** Safe localStorage helpers — never throw (e.g. private browsing, quota exceeded). */
function safeGet(key: string): string | null {
  try { return localStorage.getItem(key) } catch { return null }
}
function safeSet(key: string, value: string): void {
  try { localStorage.setItem(key, value) } catch { /* quota exceeded or private mode */ }
}
function safeRemove(key: string): void {
  try { localStorage.removeItem(key) } catch { /* ignore */ }
}

export const useStore = create<AppState>((set) => ({
  // Auth
  token: safeGet('sv_token'),
  user: (() => {
    try {
      const raw = safeGet('sv_user')
      return raw ? JSON.parse(raw) : null
    } catch {
      return null
    }
  })(),
  lastActivity: Date.now(),
  setAuth: (token, user) => {
    safeSet('sv_token', token)
    safeSet('sv_user', JSON.stringify(user))
    set({ token, user, lastActivity: Date.now() })
  },
  logout: () => {
    safeRemove('sv_token')
    safeRemove('sv_user')
    set({ token: null, user: null })
  },
  updateActivity: () => set({ lastActivity: Date.now() }),

  // Patients
  patients: [],
  setPatients: (patients) => set({ patients }),
  updatePatient: (id, update) =>
    set((s) => ({
      patients: s.patients.map((p) => (p.id === id ? { ...p, ...update } : p)),
    })),

  // Alerts
  alerts: [],
  addAlert: (alert) =>
    set((s) => ({ alerts: [alert, ...s.alerts].slice(0, 100) })),
  dismissAlert: (id) =>
    set((s) => ({
      alerts: s.alerts.map((a) => (a.id === id ? { ...a, dismissed: true } : a)),
    })),
  acknowledgeAlert: (id) =>
    set((s) => ({
      alerts: s.alerts.map((a) => (a.id === id ? { ...a, acknowledged: true } : a)),
    })),
  setAlertNote: (id, note) =>
    set((s) => ({
      alerts: s.alerts.map((a) => (a.id === id ? { ...a, note } : a)),
    })),
  clearAlerts: () => set({ alerts: [] }),

  // WebSocket
  wsConnected: false,
  setWsConnected: (connected) => set({ wsConnected: connected }),
  wsState: 'offline' as const,
  setWsState: (state) => set({ wsState: state }),

  // UI
  sidebarOpen: false,
  setSidebarOpen: (open) => set({ sidebarOpen: open }),
  showSessionWarning: false,
  setShowSessionWarning: (show) => set({ showSessionWarning: show }),

  // Monitor
  monitoredPatients: {},
  setMonitoredPatients: (patients) =>
    set({
      monitoredPatients: Object.fromEntries(
        patients.map((p) => [p.patient_id, p])
      ),
    }),
  updatePatientRisk: (patientId, update) =>
    set((s) => {
      const existing = s.monitoredPatients[patientId]
      if (!existing) return s
      const updated = { ...existing, ...update }
      // Append to risk_history (keep last 1440 entries = 24h at 1/min)
      if (update.risk_probability !== undefined) {
        updated.risk_history = [
          ...(existing.risk_history || []),
          { timestamp: Date.now() / 1000, risk_probability: update.risk_probability },
        ].slice(-1440)
      }
      return {
        monitoredPatients: { ...s.monitoredPatients, [patientId]: updated },
      }
    }),
  removeMonitoredPatient: (patientId) =>
    set((s) => {
      const { [patientId]: _, ...rest } = s.monitoredPatients
      return { monitoredPatients: rest }
    }),

  // Simulator
  simulatorSessions: [],
  addSimSession: (session) =>
    set((s) => ({ simulatorSessions: [...s.simulatorSessions, session] })),
  removeSimSession: (sessionId) =>
    set((s) => ({
      simulatorSessions: s.simulatorSessions.filter((ss) => ss.session_id !== sessionId),
    })),
  updateSimSession: (sessionId, update) =>
    set((s) => ({
      simulatorSessions: s.simulatorSessions.map((ss) =>
        ss.session_id === sessionId ? { ...ss, ...update } : ss,
      ),
    })),
  simulatorEnabled: false,
  setSimulatorEnabled: (enabled) => set({ simulatorEnabled: enabled }),
}))
