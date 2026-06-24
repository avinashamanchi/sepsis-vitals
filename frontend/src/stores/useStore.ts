import { create } from 'zustand'
import type { Alert, Patient } from '../types'

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

  // UI
  sidebarOpen: boolean
  setSidebarOpen: (open: boolean) => void
}

export const useStore = create<AppState>((set) => ({
  // Auth
  token: localStorage.getItem('sv_token'),
  user: (() => {
    try {
      const raw = localStorage.getItem('sv_user')
      return raw ? JSON.parse(raw) : null
    } catch {
      return null
    }
  })(),
  lastActivity: Date.now(),
  setAuth: (token, user) => {
    localStorage.setItem('sv_token', token)
    localStorage.setItem('sv_user', JSON.stringify(user))
    set({ token, user, lastActivity: Date.now() })
  },
  logout: () => {
    localStorage.removeItem('sv_token')
    localStorage.removeItem('sv_user')
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

  // UI
  sidebarOpen: false,
  setSidebarOpen: (open) => set({ sidebarOpen: open }),
}))
