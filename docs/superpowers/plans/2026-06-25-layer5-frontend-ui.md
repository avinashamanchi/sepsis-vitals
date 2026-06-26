# Layer 5: Frontend UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Three new/overhauled pages (Monitor, PatientDetail, Predict) connected to the autonomous prediction engine, plus a simulator control panel -- designed for clinical workflows with fixed patient order, 24-hour sparklines, and real-time WebSocket updates.

**Architecture:** Foundation-first approach: types -> API client -> store -> WebSocket -> navigation -> pages/components. Each page is a self-contained React component using Zustand for state, Recharts for charts, and Lucide for icons. The Monitor page is the centerpiece -- a real-time ward display with patient cards that update via WebSocket without reordering.

**Tech Stack:** React 19, TypeScript 6, Tailwind 4 (CSS-only theming via `@theme`), Recharts 3, Zustand 5, React Router 7, Lucide React, Vite 8. No test runner is configured -- verification is via `tsc -b` (type checking) and `vite build` (full build).

---

### File Structure

```
frontend/src/
  types/index.ts              -- MODIFY: add MonitoredPatient, SimSession, ClinicalScores, DeteriorationAlert
  lib/api.ts                  -- MODIFY: add monitor + simulator API endpoints
  stores/useStore.ts          -- MODIFY: add monitoredPatients, simulatorSessions state
  hooks/useWebSocket.ts       -- MODIFY: handle patient_update, deterioration_alert, recovery_alert
  App.tsx                     -- MODIFY: add /monitor route
  components/Sidebar.tsx      -- MODIFY: add Monitor nav entry
  components/BottomNav.tsx    -- MODIFY: add Monitor nav entry
  components/TrendArrow.tsx   -- CREATE: reusable trend direction indicator
  pages/Monitor.tsx           -- CREATE: real-time ward monitoring page
  components/SimulatorPanel.tsx -- CREATE: floating simulator control panel
  pages/PatientDetail.tsx     -- MODIFY: overhaul with clinical scores, CI band, feature importance
  pages/Predict.tsx           -- MODIFY: add demographics, comorbidities, dual thresholds, auto-monitor
```

---

### Task 1: Types — MonitoredPatient, SimSession, ClinicalScores, DeteriorationAlert

**Files:**
- Modify: `frontend/src/types/index.ts`

- [ ] **Step 1: Add new types to types/index.ts**

Append after the existing `ScoreResult` interface (after line 61):

```typescript
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
```

- [ ] **Step 2: Verify types compile**

Run: `cd frontend && npx tsc --noEmit`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add frontend/src/types/index.ts
git commit -m "feat(frontend): add MonitoredPatient, SimSession, ClinicalScores, DeteriorationAlert types"
```

---

### Task 2: API Client — Monitor and Simulator Endpoints

**Files:**
- Modify: `frontend/src/lib/api.ts`

The backend already has these endpoints (from Layers 3-4):
- `POST /monitor/register` — body: `{ patient_id, demographics?, comorbidities? }`
- `DELETE /monitor/{patient_id}` — unregister from monitoring
- `GET /monitor/status` — returns `{ patients: [...], count }`
- `POST /simulator/ward` — body: `{ n_patients?, speed?, sepsis_count?, seed? }`
- `POST /simulator/replay` — body: `{ subject_id?, speed?, sepsis_only? }`
- `DELETE /simulator/{session_id}` — stop session
- `GET /simulator/sessions` — returns `{ sessions: [...] }`
- `GET /simulator/cases` — returns `{ cases: [...], count }`

- [ ] **Step 1: Add monitor and simulator methods to the api object**

In `frontend/src/lib/api.ts`, add these methods inside the `api` object (after the `systemHealth` method, before the closing `}`):

```typescript
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
```

- [ ] **Step 2: Verify it compiles**

Run: `cd frontend && npx tsc --noEmit`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add frontend/src/lib/api.ts
git commit -m "feat(frontend): add monitor and simulator API client methods"
```

---

### Task 3: Store — monitoredPatients and simulatorSessions State

**Files:**
- Modify: `frontend/src/stores/useStore.ts`

- [ ] **Step 1: Add imports for new types**

At the top of `frontend/src/stores/useStore.ts`, change the import line:

```typescript
import type { Alert, Patient, MonitoredPatient, SimSession } from '../types'
```

- [ ] **Step 2: Add new state and actions to the AppState interface**

Add these after the `setSidebarOpen` line (line 32) in the interface:

```typescript
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
```

- [ ] **Step 3: Add the state implementations**

Add these after the `setSidebarOpen` implementation (line 91) in the `create` callback:

```typescript
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
```

- [ ] **Step 4: Verify it compiles**

Run: `cd frontend && npx tsc --noEmit`
Expected: No errors

- [ ] **Step 5: Commit**

```bash
git add frontend/src/stores/useStore.ts
git commit -m "feat(frontend): add monitoredPatients and simulatorSessions store state"
```

---

### Task 4: WebSocket — Handle patient_update, deterioration_alert, recovery_alert

**Files:**
- Modify: `frontend/src/hooks/useWebSocket.ts`

The backend sends these WebSocket message types (from Layer 3's VitalsIngester):
- `patient_update`: `{ type, patient_id, risk_level, risk_probability, timestamp, trend }`
- `deterioration_alert`: `{ type, patient_id, risk_level, risk_probability, alert_type, risk_delta, deterioration_rate, window_hours, previous_risk_level, timestamp }`
- `recovery_alert`: `{ type, patient_id, risk_level, risk_probability, alert_type, risk_delta, previous_risk_level, timestamp }`
- `sepsis_alert`: already handled (existing code)

- [ ] **Step 1: Add store selectors for new actions**

In `frontend/src/hooks/useWebSocket.ts`, add `updatePatientRisk` to the store selectors. Change lines 24-26 to:

```typescript
  const setWsConnected = useStore((s) => s.setWsConnected)
  const addAlert = useStore((s) => s.addAlert)
  const updatePatientRisk = useStore((s) => s.updatePatientRisk)
```

- [ ] **Step 2: Add handlers for new message types**

In `frontend/src/hooks/useWebSocket.ts`, inside the `ws.onmessage` handler (after the `if (data.type === 'sepsis_alert')` block's closing brace on line 72, before the closing `} catch`), add:

```typescript
            if (data.type === 'patient_update' || data.type === 'deterioration_alert' || data.type === 'recovery_alert') {
              updatePatientRisk(data.patient_id, {
                risk_probability: data.risk_probability ?? 0,
                risk_level: data.risk_level ?? 'low',
                last_prediction_time: Date.now() / 1000,
              })
            }

            if (data.type === 'deterioration_alert') {
              const riskLevel = data.risk_level ?? 'high'
              addAlert({
                id: `ws-det-${Date.now()}-${crypto.randomUUID?.() ?? Math.random().toString(36).slice(2, 10)}`,
                patientId: data.patient_id ?? '',
                riskLevel,
                riskProbability: data.risk_probability ?? 0,
                message: `Deterioration: risk increased by ${((data.risk_delta ?? 0) * 100).toFixed(0)}% over ${(data.window_hours ?? 0).toFixed(1)}h`,
                timestamp: data.timestamp ?? new Date().toISOString(),
                dismissed: false,
              })
              playAlertSound(riskLevel)
            }

            if (data.type === 'recovery_alert') {
              addAlert({
                id: `ws-rec-${Date.now()}-${crypto.randomUUID?.() ?? Math.random().toString(36).slice(2, 10)}`,
                patientId: data.patient_id ?? '',
                riskLevel: data.risk_level ?? 'low',
                riskProbability: data.risk_probability ?? 0,
                message: `Recovery: risk decreased by ${(Math.abs(data.risk_delta ?? 0) * 100).toFixed(0)}%`,
                timestamp: data.timestamp ?? new Date().toISOString(),
                dismissed: false,
              })
            }
```

- [ ] **Step 3: Add updatePatientRisk to the useEffect dependency array**

Change line 111:

```typescript
  }, [setWsConnected, addAlert, updatePatientRisk])
```

- [ ] **Step 4: Verify it compiles**

Run: `cd frontend && npx tsc --noEmit`
Expected: No errors

- [ ] **Step 5: Commit**

```bash
git add frontend/src/hooks/useWebSocket.ts
git commit -m "feat(frontend): handle patient_update, deterioration_alert, recovery_alert WebSocket messages"
```

---

### Task 5: Navigation — Monitor Route, Sidebar, BottomNav

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/components/Sidebar.tsx`
- Modify: `frontend/src/components/BottomNav.tsx`

- [ ] **Step 1: Add Monitor lazy import to App.tsx**

In `frontend/src/App.tsx`, add after line 18 (the `Login` lazy import):

```typescript
const Monitor = lazy(() => import('./pages/Monitor').then((m) => ({ default: m.Monitor })))
```

- [ ] **Step 2: Add /monitor route to App.tsx**

In `frontend/src/App.tsx`, add the Monitor route after the `/patients/:id` route (line 115):

```typescript
                          <Route path="/monitor" element={<Monitor />} />
```

- [ ] **Step 3: Add Monitor entry to Sidebar.tsx NAV_ITEMS**

In `frontend/src/components/Sidebar.tsx`, add the Monitor entry to NAV_ITEMS. The icon `Activity` is already imported. Add after the Patients entry (line 11) so Monitor appears after Patients:

```typescript
  { to: '/monitor', icon: Activity, label: 'Monitor' },
```

Note: `Activity` is already imported on line 4. However, it's currently used by the Alerts entry. To avoid icon duplication, use a different icon for Alerts. Change the import on line 4 to add `MonitorDot`:

```typescript
import {
  BarChart3, Brain, Calculator,
  LayoutDashboard, LogOut, MonitorDot, Settings, Shield, Users, Wifi, WifiOff,
} from 'lucide-react'
```

And update NAV_ITEMS to use `Activity` for Monitor and a different icon for Alerts. The full NAV_ITEMS becomes:

```typescript
const NAV_ITEMS = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/patients', icon: Users, label: 'Patients' },
  { to: '/monitor', icon: MonitorDot, label: 'Monitor' },
  { to: '/scores', icon: Calculator, label: 'Score Lab' },
  { to: '/predict', icon: Brain, label: 'AI Predict' },
  { to: '/analytics', icon: BarChart3, label: 'Analytics' },
  { to: '/alerts', icon: Activity, label: 'Alerts' },
  { to: '/admin', icon: Settings, label: 'Admin' },
]
```

Wait -- the spec says to use `Activity` icon for Monitor. Let's keep it simple and follow the spec. Use `Activity` for Monitor and `Bell` for Alerts:

```typescript
import {
  Activity, BarChart3, Bell, Brain, Calculator,
  LayoutDashboard, LogOut, Settings, Shield, Users, Wifi, WifiOff,
} from 'lucide-react'
```

```typescript
const NAV_ITEMS = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/patients', icon: Users, label: 'Patients' },
  { to: '/monitor', icon: Activity, label: 'Monitor' },
  { to: '/scores', icon: Calculator, label: 'Score Lab' },
  { to: '/predict', icon: Brain, label: 'AI Predict' },
  { to: '/analytics', icon: BarChart3, label: 'Analytics' },
  { to: '/alerts', icon: Bell, label: 'Alerts' },
  { to: '/admin', icon: Settings, label: 'Admin' },
]
```

- [ ] **Step 4: Add Monitor entry to BottomNav.tsx NAV_ITEMS**

In `frontend/src/components/BottomNav.tsx`, update the import and NAV_ITEMS:

```typescript
import {
  Activity, BarChart3, Bell, Brain,
  LayoutDashboard, Settings, Users,
} from 'lucide-react'
```

```typescript
const NAV_ITEMS = [
  { to: '/', icon: LayoutDashboard, label: 'Home' },
  { to: '/patients', icon: Users, label: 'Patients' },
  { to: '/monitor', icon: Activity, label: 'Monitor' },
  { to: '/predict', icon: Brain, label: 'Predict' },
  { to: '/alerts', icon: Bell, label: 'Alerts' },
  { to: '/admin', icon: Settings, label: 'Admin' },
]
```

- [ ] **Step 5: Verify it compiles**

Run: `cd frontend && npx tsc --noEmit`
Expected: Will fail because `Monitor` page doesn't exist yet. Create a placeholder:

Create `frontend/src/pages/Monitor.tsx`:

```typescript
export function Monitor() {
  return <div>Monitor page (under construction)</div>
}
```

Run: `cd frontend && npx tsc --noEmit`
Expected: No errors

- [ ] **Step 6: Commit**

```bash
git add frontend/src/App.tsx frontend/src/components/Sidebar.tsx frontend/src/components/BottomNav.tsx frontend/src/pages/Monitor.tsx
git commit -m "feat(frontend): add Monitor route and navigation entries"
```

---

### Task 6: TrendArrow Component

**Files:**
- Create: `frontend/src/components/TrendArrow.tsx`

A small reusable component used by both Monitor.tsx and PatientDetail.tsx.

- [ ] **Step 1: Create TrendArrow.tsx**

Create `frontend/src/components/TrendArrow.tsx`:

```typescript
import { TrendingUp, TrendingDown, Minus } from 'lucide-react'
import clsx from 'clsx'

interface TrendArrowProps {
  direction: 'improving' | 'stable' | 'worsening' | 'unknown'
  rateText?: string
  size?: 'sm' | 'md'
}

export function TrendArrow({ direction, rateText, size = 'sm' }: TrendArrowProps) {
  const iconSize = size === 'sm' ? 'w-3.5 h-3.5' : 'w-4 h-4'
  const textSize = size === 'sm' ? 'text-xs' : 'text-sm'

  if (direction === 'worsening') {
    return (
      <span className={clsx('inline-flex items-center gap-1 text-danger', textSize)}>
        <TrendingUp className={iconSize} />
        {rateText && <span>{rateText}</span>}
      </span>
    )
  }
  if (direction === 'improving') {
    return (
      <span className={clsx('inline-flex items-center gap-1 text-accent', textSize)}>
        <TrendingDown className={iconSize} />
        {rateText && <span>{rateText}</span>}
      </span>
    )
  }
  return (
    <span className={clsx('inline-flex items-center gap-1 text-text-muted', textSize)}>
      <Minus className={iconSize} />
      {rateText && <span>{rateText}</span>}
    </span>
  )
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cd frontend && npx tsc --noEmit`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/TrendArrow.tsx
git commit -m "feat(frontend): add TrendArrow reusable component"
```

---

### Task 7: Monitor Page — Real-Time Ward Monitoring

**Files:**
- Modify: `frontend/src/pages/Monitor.tsx` (replace placeholder)

This is the centerpiece page. Key design decisions from the spec:
- **Fixed patient order** by registration order -- never re-sorts automatically
- **24-hour sparklines** of risk scores using Recharts LineChart (minimal, no axes)
- **Risk indication via card border color** -- not sort order
- **Live WebSocket updates** update cards in-place without reordering
- **Click-through** to `/patients/:id`
- **Explicit "Sort by Risk" button** requires deliberate click

The page loads initial state from `GET /monitor/status`, then relies on WebSocket for live updates (handled by useWebSocket hook + store).

- [ ] **Step 1: Replace Monitor.tsx placeholder with full implementation**

Replace `frontend/src/pages/Monitor.tsx` entirely:

```typescript
import { useEffect, useState, useMemo } from 'react'
import { Link } from 'react-router-dom'
import { Activity, ArrowUpDown, RefreshCw } from 'lucide-react'
import { LineChart, Line, ResponsiveContainer } from 'recharts'
import { useStore } from '../stores/useStore'
import { api, isDemo } from '../lib/api'
import { RiskBadge } from '../components/RiskBadge'
import { TrendArrow } from '../components/TrendArrow'
import { LoadingSpinner } from '../components/LoadingSpinner'
import type { RiskLevel, MonitoredPatient } from '../types'
import { RISK_BORDER } from '../lib/risk'
import clsx from 'clsx'

/** Demo patients for GitHub Pages mode */
function makeDemoPatients(): MonitoredPatient[] {
  const names = ['P-1001', 'P-1002', 'P-1003', 'P-1004', 'P-1005', 'P-1006']
  const risks: Array<{ prob: number; level: RiskLevel; trend: MonitoredPatient['trend_direction'] }> = [
    { prob: 0.12, level: 'low', trend: 'stable' },
    { prob: 0.82, level: 'critical', trend: 'worsening' },
    { prob: 0.35, level: 'moderate', trend: 'stable' },
    { prob: 0.58, level: 'high', trend: 'worsening' },
    { prob: 0.08, level: 'low', trend: 'improving' },
    { prob: 0.42, level: 'moderate', trend: 'stable' },
  ]
  const now = Date.now() / 1000
  return names.map((id, i) => {
    const r = risks[i]
    const history = Array.from({ length: 48 }, (_, j) => ({
      timestamp: now - (48 - j) * 1800,
      risk_probability: Math.max(0.02, Math.min(0.98, r.prob + 0.1 * Math.sin(j * 0.5 + i))),
    }))
    return {
      patient_id: id,
      demographics: {},
      vitals: { heart_rate: 72 + i * 5, temperature: 36.5 + i * 0.3, sbp: 120 - i * 5, spo2: 98 - i },
      risk_probability: r.prob,
      risk_level: r.level,
      trend_direction: r.trend,
      last_prediction_time: now - i * 120,
      last_vitals_time: now - i * 60,
      registered_at: now - 3600 * (6 - i),
      alert_state: 'normal',
      deterioration_rate: 0,
      window_hours: 0,
      risk_history: history,
    }
  })
}

function timeAgo(unixSeconds: number): string {
  const diff = Date.now() / 1000 - unixSeconds
  if (diff < 60) return 'just now'
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`
  return `${Math.floor(diff / 86400)}d ago`
}

/** Normal range check for vital highlighting */
function isAbnormal(key: string, value: number): boolean {
  const ranges: Record<string, [number, number]> = {
    heart_rate: [60, 100],
    temperature: [36.1, 38.0],
    sbp: [90, 140],
    spo2: [95, 100],
    resp_rate: [12, 20],
  }
  const range = ranges[key]
  if (!range) return false
  return value < range[0] || value > range[1]
}

export function Monitor() {
  const monitoredPatients = useStore((s) => s.monitoredPatients)
  const setMonitoredPatients = useStore((s) => s.setMonitoredPatients)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [sortByRisk, setSortByRisk] = useState(false)

  // Load initial state from REST API
  useEffect(() => {
    if (isDemo) {
      setMonitoredPatients(makeDemoPatients())
      setLoading(false)
      return
    }
    api.monitorStatus()
      .then((data) => {
        const patients: MonitoredPatient[] = data.patients.map((p) => ({
          ...p,
          risk_level: (p.risk_level as RiskLevel) || 'low',
          trend_direction: (p.trend_direction as MonitoredPatient['trend_direction']) || 'unknown',
          alert_state: (p.alert_state as MonitoredPatient['alert_state']) || 'normal',
          deterioration_rate: p.deterioration_rate ?? 0,
          window_hours: p.window_hours ?? 0,
          risk_history: [],
        }))
        setMonitoredPatients(patients)
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : 'Failed to load monitor status'))
      .finally(() => setLoading(false))
  }, [setMonitoredPatients])

  // Stable order: sorted by registered_at (insertion order), or by risk if toggled
  const patients = useMemo(() => {
    const list = Object.values(monitoredPatients)
    if (sortByRisk) {
      const riskOrder: Record<string, number> = { critical: 0, high: 1, moderate: 2, low: 3 }
      return [...list].sort((a, b) => (riskOrder[a.risk_level] ?? 4) - (riskOrder[b.risk_level] ?? 4))
    }
    return [...list].sort((a, b) => a.registered_at - b.registered_at)
  }, [monitoredPatients, sortByRisk])

  return (
    <div className="space-y-5 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="font-heading text-2xl font-bold flex items-center gap-2">
            <Activity className="w-6 h-6 text-accent" />
            Ward Monitor
          </h1>
          <p className="text-sm text-text-secondary mt-1">
            {patients.length} patient{patients.length !== 1 ? 's' : ''} under continuous monitoring
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setSortByRisk(!sortByRisk)}
            className={clsx(
              'flex items-center gap-1.5 px-3 py-1.5 text-xs rounded border transition-colors',
              sortByRisk
                ? 'bg-accent/10 text-accent border-accent/30'
                : 'bg-elevated text-text-secondary border-border hover:text-text-primary',
            )}
          >
            <ArrowUpDown className="w-3.5 h-3.5" />
            Sort by Risk
          </button>
          <button
            onClick={() => {
              setLoading(true)
              api.monitorStatus()
                .then((data) => {
                  const pts: MonitoredPatient[] = data.patients.map((p) => ({
                    ...p,
                    risk_level: (p.risk_level as RiskLevel) || 'low',
                    trend_direction: (p.trend_direction as MonitoredPatient['trend_direction']) || 'unknown',
                    alert_state: (p.alert_state as MonitoredPatient['alert_state']) || 'normal',
                    deterioration_rate: p.deterioration_rate ?? 0,
                    window_hours: p.window_hours ?? 0,
                    risk_history: [],
                  }))
                  setMonitoredPatients(pts)
                })
                .catch(() => {})
                .finally(() => setLoading(false))
            }}
            className="p-1.5 text-text-muted hover:text-text-primary transition-colors"
            title="Refresh"
          >
            <RefreshCw className={clsx('w-4 h-4', loading && 'animate-spin')} />
          </button>
        </div>
      </div>

      {loading && patients.length === 0 && (
        <LoadingSpinner size="lg" label="Loading monitor..." className="py-12" />
      )}

      {error && (
        <div className="bg-danger/10 border border-danger/20 rounded-lg p-4 text-sm text-danger">
          {error}
        </div>
      )}

      {/* Patient Grid */}
      {patients.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4 gap-4">
          {patients.map((patient) => (
            <Link
              key={patient.patient_id}
              to={`/patients/${patient.patient_id}`}
              className={clsx(
                'bg-surface border-2 rounded-lg p-4 transition-all hover:bg-elevated block',
                RISK_BORDER[patient.risk_level],
                patient.risk_level === 'critical' && 'animate-pulse-critical',
              )}
            >
              {/* Header row */}
              <div className="flex items-center justify-between mb-3">
                <span className="font-heading font-semibold text-sm">{patient.patient_id}</span>
                <RiskBadge level={patient.risk_level} size="sm" pulse={patient.risk_level === 'critical'} />
              </div>

              {/* Risk probability + trend */}
              <div className="flex items-center justify-between mb-3">
                <span className="text-2xl font-bold font-heading">
                  {(patient.risk_probability * 100).toFixed(0)}%
                </span>
                <TrendArrow
                  direction={patient.trend_direction}
                  rateText={
                    patient.deterioration_rate
                      ? `${patient.deterioration_rate > 0 ? '+' : ''}${(patient.deterioration_rate * 100).toFixed(0)}%/h`
                      : undefined
                  }
                />
              </div>

              {/* 24h Sparkline */}
              {patient.risk_history && patient.risk_history.length > 1 && (
                <div className="h-[40px] mb-3">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={patient.risk_history}>
                      <Line
                        type="monotone"
                        dataKey="risk_probability"
                        stroke={
                          patient.risk_level === 'critical' ? '#ff3b5c' :
                          patient.risk_level === 'high' ? '#ff6b35' :
                          patient.risk_level === 'moderate' ? '#ffb830' : '#00ff9d'
                        }
                        strokeWidth={1.5}
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}

              {/* Vitals snapshot */}
              <div className="grid grid-cols-4 gap-1 text-[10px] mb-2">
                {(['heart_rate', 'temperature', 'sbp', 'spo2'] as const).map((key) => {
                  const value = patient.vitals[key]
                  if (value == null) return null
                  const labels: Record<string, string> = { heart_rate: 'HR', temperature: 'Temp', sbp: 'SBP', spo2: 'SpO2' }
                  const units: Record<string, string> = { heart_rate: 'bpm', temperature: '°C', sbp: 'mmHg', spo2: '%' }
                  return (
                    <div key={key} className="text-center">
                      <div className="text-text-muted">{labels[key]}</div>
                      <div className={clsx('font-medium', isAbnormal(key, value) ? 'text-danger' : 'text-text-primary')}>
                        {key === 'temperature' ? value.toFixed(1) : Math.round(value)}
                        <span className="text-text-muted ml-0.5">{units[key]}</span>
                      </div>
                    </div>
                  )
                })}
              </div>

              {/* Last updated */}
              <div className="text-[10px] text-text-muted text-right">
                {timeAgo(patient.last_vitals_time || patient.last_prediction_time)}
              </div>
            </Link>
          ))}
        </div>
      )}

      {!loading && patients.length === 0 && !error && (
        <div className="bg-surface border border-border rounded-lg p-8 text-center">
          <Activity className="w-8 h-8 text-text-muted mx-auto mb-3" />
          <p className="text-sm text-text-muted">No patients under monitoring</p>
          <p className="text-xs text-text-muted mt-1">Register patients via the API or start a simulation</p>
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cd frontend && npx tsc --noEmit`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add frontend/src/pages/Monitor.tsx
git commit -m "feat(frontend): implement Monitor page with patient grid, sparklines, and fixed order"
```

---

### Task 8: SimulatorPanel — Floating Control Panel

**Files:**
- Create: `frontend/src/components/SimulatorPanel.tsx`
- Modify: `frontend/src/App.tsx` (render the panel)

The SimulatorPanel is a floating, collapsible panel (bottom-right) that provides controls for:
- **Replay tab:** Select a MIMIC-IV case and replay it
- **Ward sim tab:** Configure and start a synthetic ward simulation
- **Active sessions:** List with stop buttons
- Only rendered when `simulatorEnabled` is `true` in store

- [ ] **Step 1: Create SimulatorPanel.tsx**

Create `frontend/src/components/SimulatorPanel.tsx`:

```typescript
import { useState, useEffect } from 'react'
import { Play, Square, ChevronDown, ChevronUp, FlaskConical } from 'lucide-react'
import { useStore } from '../stores/useStore'
import { api } from '../lib/api'
import clsx from 'clsx'

type Tab = 'replay' | 'ward'

interface MIMICCase {
  subject_id: number
  age_years: number
  sex: string
  sepsis_label: number
  icu_los_hours: number
  n_observations: number
}

export function SimulatorPanel() {
  const simulatorEnabled = useStore((s) => s.simulatorEnabled)
  const sessions = useStore((s) => s.simulatorSessions)
  const addSimSession = useStore((s) => s.addSimSession)
  const removeSimSession = useStore((s) => s.removeSimSession)

  const [collapsed, setCollapsed] = useState(true)
  const [tab, setTab] = useState<Tab>('ward')
  const [cases, setCases] = useState<MIMICCase[]>([])
  const [loading, setLoading] = useState(false)

  // Ward sim controls
  const [wardPatients, setWardPatients] = useState(8)
  const [wardSpeed, setWardSpeed] = useState(360)
  const [wardSepsis, setWardSepsis] = useState(2)

  // Replay controls
  const [selectedCase, setSelectedCase] = useState<number | null>(null)
  const [replaySpeed, setReplaySpeed] = useState(720)

  // Load cases when panel opens
  useEffect(() => {
    if (!collapsed && simulatorEnabled && cases.length === 0) {
      api.simulatorCases()
        .then((data) => setCases(data.cases))
        .catch(() => {})
    }
  }, [collapsed, simulatorEnabled, cases.length])

  if (!simulatorEnabled) return null

  const startWard = async () => {
    setLoading(true)
    try {
      const res = await api.simulatorStartWard({
        n_patients: wardPatients,
        speed: wardSpeed,
        sepsis_count: wardSepsis,
      })
      addSimSession({
        session_id: res.session_id,
        type: 'ward',
        status: 'running',
        patient_count: wardPatients,
        started_at: Date.now() / 1000,
      })
    } catch { /* ignore */ }
    setLoading(false)
  }

  const startReplay = async () => {
    if (!selectedCase) return
    setLoading(true)
    try {
      const res = await api.simulatorStartReplay({
        subject_id: selectedCase,
        speed: replaySpeed,
      })
      addSimSession({
        session_id: res.session_id,
        type: 'replay',
        status: 'running',
        subject_id: res.subject_id,
        started_at: Date.now() / 1000,
      })
    } catch { /* ignore */ }
    setLoading(false)
  }

  const stopSession = async (sessionId: string) => {
    try {
      await api.simulatorStop(sessionId)
      removeSimSession(sessionId)
    } catch { /* ignore */ }
  }

  return (
    <div className="fixed bottom-4 right-4 z-50 w-80">
      {/* Toggle bar */}
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="w-full flex items-center justify-between bg-overlay border border-border rounded-t-lg px-4 py-2 text-sm font-medium text-text-primary hover:bg-elevated transition-colors"
      >
        <span className="flex items-center gap-2">
          <FlaskConical className="w-4 h-4 text-info" />
          Simulator
          {sessions.length > 0 && (
            <span className="bg-accent/20 text-accent text-xs px-1.5 py-0.5 rounded-full">
              {sessions.length}
            </span>
          )}
        </span>
        {collapsed ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
      </button>

      {/* Panel body */}
      {!collapsed && (
        <div className="bg-surface border border-t-0 border-border rounded-b-lg p-4 space-y-4 max-h-[400px] overflow-y-auto">
          {/* Tabs */}
          <div className="flex gap-1 bg-elevated rounded p-0.5">
            {(['ward', 'replay'] as const).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={clsx(
                  'flex-1 text-xs py-1.5 rounded transition-colors capitalize',
                  tab === t ? 'bg-accent/10 text-accent' : 'text-text-muted hover:text-text-secondary',
                )}
              >
                {t === 'ward' ? 'Ward Sim' : 'Case Replay'}
              </button>
            ))}
          </div>

          {/* Ward Tab */}
          {tab === 'ward' && (
            <div className="space-y-3">
              <div>
                <label className="text-[10px] text-text-muted">Patients: {wardPatients}</label>
                <input
                  type="range"
                  min={4}
                  max={20}
                  value={wardPatients}
                  onChange={(e) => setWardPatients(Number(e.target.value))}
                  className="w-full accent-accent"
                />
              </div>
              <div>
                <label className="text-[10px] text-text-muted">Speed: {wardSpeed}x</label>
                <input
                  type="range"
                  min={1}
                  max={1000}
                  step={10}
                  value={wardSpeed}
                  onChange={(e) => setWardSpeed(Number(e.target.value))}
                  className="w-full accent-accent"
                />
              </div>
              <div>
                <label className="text-[10px] text-text-muted">Sepsis patients: {wardSepsis}</label>
                <input
                  type="range"
                  min={0}
                  max={Math.floor(wardPatients / 2)}
                  value={wardSepsis}
                  onChange={(e) => setWardSepsis(Number(e.target.value))}
                  className="w-full accent-accent"
                />
              </div>
              <button
                onClick={startWard}
                disabled={loading}
                className="w-full flex items-center justify-center gap-1.5 bg-accent/10 text-accent border border-accent/30 rounded py-2 text-xs font-medium hover:bg-accent/20 disabled:opacity-50"
              >
                <Play className="w-3.5 h-3.5" />
                Start Ward Simulation
              </button>
            </div>
          )}

          {/* Replay Tab */}
          {tab === 'replay' && (
            <div className="space-y-3">
              <select
                value={selectedCase ?? ''}
                onChange={(e) => setSelectedCase(e.target.value ? Number(e.target.value) : null)}
                className="w-full bg-elevated border border-border rounded px-3 py-2 text-xs text-text-primary"
              >
                <option value="">Select a case...</option>
                {cases.map((c) => (
                  <option key={c.subject_id} value={c.subject_id}>
                    #{c.subject_id} | {c.sex} {c.age_years}y | {c.sepsis_label ? 'SEPSIS' : 'No sepsis'} | {(c.icu_los_hours / 24).toFixed(1)}d
                  </option>
                ))}
              </select>
              <div>
                <label className="text-[10px] text-text-muted">Speed: {replaySpeed}x</label>
                <input
                  type="range"
                  min={1}
                  max={1000}
                  step={10}
                  value={replaySpeed}
                  onChange={(e) => setReplaySpeed(Number(e.target.value))}
                  className="w-full accent-accent"
                />
              </div>
              <button
                onClick={startReplay}
                disabled={loading || !selectedCase}
                className="w-full flex items-center justify-center gap-1.5 bg-accent/10 text-accent border border-accent/30 rounded py-2 text-xs font-medium hover:bg-accent/20 disabled:opacity-50"
              >
                <Play className="w-3.5 h-3.5" />
                Start Replay
              </button>
            </div>
          )}

          {/* Active Sessions */}
          {sessions.length > 0 && (
            <div className="space-y-2">
              <h3 className="text-[10px] text-text-muted uppercase tracking-wider">Active Sessions</h3>
              {sessions.map((s) => (
                <div
                  key={s.session_id}
                  className="flex items-center justify-between bg-elevated rounded p-2"
                >
                  <div className="text-xs">
                    <span className="text-text-primary capitalize">{s.type}</span>
                    <span className="text-text-muted ml-2">{s.session_id.slice(0, 8)}</span>
                  </div>
                  <button
                    onClick={() => stopSession(s.session_id)}
                    className="p-1 text-text-muted hover:text-danger transition-colors"
                    title="Stop"
                  >
                    <Square className="w-3.5 h-3.5" />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 2: Render SimulatorPanel in App.tsx**

In `frontend/src/App.tsx`, import the SimulatorPanel at the top:

```typescript
import { SimulatorPanel } from './components/SimulatorPanel'
```

Then add it inside the AuthGuard div, right after `<BottomNav />` (line 126):

```typescript
                  <SimulatorPanel />
```

- [ ] **Step 3: Check simulator_enabled on app load**

We need to detect if the simulator is enabled. The simplest way: attempt to call `GET /simulator/sessions`. If it returns 403, simulator is not enabled. Add a check in App.tsx.

Actually, let's keep it simpler. In `frontend/src/App.tsx`, add a useEffect at the top of the App component to check simulator status:

```typescript
  useEffect(() => {
    if (isDemo) return
    api.simulatorSessions()
      .then(() => useStore.getState().setSimulatorEnabled(true))
      .catch(() => useStore.getState().setSimulatorEnabled(false))
  }, [])
```

Add this right after the `useWebSocket()` call in the `App` function.

- [ ] **Step 4: Verify it compiles**

Run: `cd frontend && npx tsc --noEmit`
Expected: No errors

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/SimulatorPanel.tsx frontend/src/App.tsx
git commit -m "feat(frontend): add SimulatorPanel floating control panel"
```

---

### Task 9: PatientDetail Overhaul — Clinical Scores, CI Band, Feature Importance

**Files:**
- Modify: `frontend/src/pages/PatientDetail.tsx`

Overhaul the existing PatientDetail page to add:
1. **Risk timeline** with confidence interval band (area chart) and threshold lines
2. **Clinical scores panel** (qSOFA, NEWS2, SIRS, Shock Index) as colored number cards
3. **Feature importance chart** (horizontal bar chart from prediction data)
4. **Deterioration indicator** with TrendArrow
5. Keep existing vitals snapshot and vitals history charts

The page fetches data from `GET /patient/:id/trend` (existing) and optionally from `GET /monitor/status` for real-time trend data. For predictions with feature importance, it uses the Prediction type which already has `top_risk_factors` and `clinical_scores`.

- [ ] **Step 1: Replace PatientDetail.tsx with overhauled version**

Replace `frontend/src/pages/PatientDetail.tsx` entirely:

```typescript
import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { ArrowLeft, Activity, TrendingUp } from 'lucide-react'
import { api, isDemo } from '../lib/api'
import { RiskBadge } from '../components/RiskBadge'
import { TrendArrow } from '../components/TrendArrow'
import { LoadingSpinner } from '../components/LoadingSpinner'
import { useStore } from '../stores/useStore'
import type { RiskLevel } from '../types'
import clsx from 'clsx'
import {
  AreaChart, Area, LineChart, Line, BarChart, Bar, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceLine,
} from 'recharts'

interface TrendPoint {
  timestamp: string
  risk_probability: number
  vitals: Record<string, number>
}

/** Generate deterministic demo data from a patient ID. */
function makeDemoTrend(id: string): TrendPoint[] {
  const seed = id.split('').reduce((acc, c) => acc + c.charCodeAt(0), 0)
  return Array.from({ length: 24 }, (_, i) => {
    const base = 0.3 + 0.4 * Math.sin((seed + i) * 0.7)
    const prob = Math.min(0.99, Math.max(0.02, base + 0.08 * Math.sin(i * 1.3)))
    return {
      timestamp: `${String(i).padStart(2, '0')}:00`,
      risk_probability: Math.round(prob * 100) / 100,
      vitals: {
        heart_rate: 70 + Math.round(30 * Math.sin((seed + i) * 0.5)),
        temperature: 36.5 + Math.round(20 * Math.sin((seed + i) * 0.4)) / 10,
        sbp: 110 + Math.round(25 * Math.sin((seed + i) * 0.6)),
        spo2: 97 - Math.round(4 * Math.max(0, Math.sin((seed + i) * 0.8))),
        resp_rate: 16 + Math.round(8 * Math.sin((seed + i) * 0.55)),
      },
    }
  })
}

function riskFromProb(p: number): RiskLevel {
  if (p >= 0.75) return 'critical'
  if (p >= 0.50) return 'high'
  if (p >= 0.25) return 'moderate'
  return 'low'
}

/** Compute clinical scores from vitals */
function computeScores(vitals: Record<string, number>): {
  qsofa: number
  sirs: number
  news2: number
  shockIndex: number | null
} {
  let qsofa = 0
  if ((vitals.sbp ?? 999) <= 100) qsofa++
  if ((vitals.resp_rate ?? 0) >= 22) qsofa++
  if ((vitals.gcs ?? 15) < 15) qsofa++

  let sirs = 0
  if ((vitals.temperature ?? 37) > 38.0 || (vitals.temperature ?? 37) < 36.0) sirs++
  if ((vitals.heart_rate ?? 70) > 90) sirs++
  if ((vitals.resp_rate ?? 16) > 20) sirs++
  if ((vitals.wbc ?? 8) > 12 || (vitals.wbc ?? 8) < 4) sirs++

  let news2 = 0
  const rr = vitals.resp_rate ?? 16
  if (rr <= 8) news2 += 3; else if (rr <= 11) news2 += 1; else if (rr <= 20) news2 += 0; else if (rr <= 24) news2 += 2; else news2 += 3
  const spo2 = vitals.spo2 ?? 98
  if (spo2 <= 91) news2 += 3; else if (spo2 <= 93) news2 += 2; else if (spo2 <= 95) news2 += 1
  const sbp = vitals.sbp ?? 120
  if (sbp <= 90) news2 += 3; else if (sbp <= 100) news2 += 2; else if (sbp <= 110) news2 += 1; else if (sbp >= 220) news2 += 3
  const hr = vitals.heart_rate ?? 75
  if (hr <= 40) news2 += 3; else if (hr <= 50) news2 += 1; else if (hr <= 90) news2 += 0; else if (hr <= 110) news2 += 1; else if (hr <= 130) news2 += 2; else news2 += 3
  const temp = vitals.temperature ?? 37
  if (temp <= 35.0) news2 += 3; else if (temp <= 36.0) news2 += 1; else if (temp <= 38.0) news2 += 0; else if (temp <= 39.0) news2 += 1; else news2 += 2

  const shockIndex = vitals.heart_rate && vitals.sbp ? vitals.heart_rate / vitals.sbp : null

  return { qsofa, sirs, news2, shockIndex }
}

/** Score card color */
function scoreColor(label: string, value: number): string {
  if (label === 'qSOFA' && value >= 2) return 'text-danger'
  if (label === 'SIRS' && value >= 2) return 'text-warning'
  if (label === 'NEWS2' && value >= 7) return 'text-danger'
  if (label === 'NEWS2' && value >= 5) return 'text-warning'
  if (label === 'Shock Index' && value > 1.0) return 'text-danger'
  if (label === 'Shock Index' && value > 0.7) return 'text-warning'
  return 'text-accent'
}

const tooltipStyle = {
  background: '#0a1120',
  border: '1px solid rgba(255,255,255,0.06)',
  borderRadius: 8,
  color: '#e8f4ff',
  fontSize: 12,
}

const featureColors: Record<string, string> = {
  procalcitonin: '#ff3b5c',
  lactate: '#ff6b35',
  temperature: '#ffb830',
  heart_rate: '#38b4ff',
  wbc: '#00ff9d',
  resp_rate: '#38b4ff',
  sbp: '#ffb830',
  spo2: '#00ff9d',
}

export function PatientDetail() {
  const { id } = useParams<{ id: string }>()
  const [trend, setTrend] = useState<TrendPoint[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const monitoredPatient = useStore((s) => id ? s.monitoredPatients[id] : undefined)

  useEffect(() => {
    if (!id) return
    if (isDemo) {
      setTrend(makeDemoTrend(id))
      setLoading(false)
      return
    }
    api.patientTrend(id)
      .then((data) => setTrend(data.trend ?? []))
      .catch((err: unknown) => setError(err instanceof Error ? err.message : 'Failed to load patient data'))
      .finally(() => setLoading(false))
  }, [id])

  const latest = trend.length > 0 ? trend[trend.length - 1] : null
  const currentRisk: RiskLevel = latest ? riskFromProb(latest.risk_probability) : 'low'
  const scores = latest ? computeScores(latest.vitals) : null

  // Determine trend direction from monitored state or from data
  const trendDirection = monitoredPatient?.trend_direction ?? (
    trend.length >= 2
      ? (trend[trend.length - 1].risk_probability > trend[trend.length - 2].risk_probability + 0.05
        ? 'worsening'
        : trend[trend.length - 1].risk_probability < trend[trend.length - 2].risk_probability - 0.05
          ? 'improving'
          : 'stable')
      : 'unknown'
  )

  const detRate = monitoredPatient?.deterioration_rate ?? 0

  const chartData = trend.map((t) => ({
    time: t.timestamp,
    risk: Math.round(t.risk_probability * 100),
    riskUpper: Math.min(100, Math.round(t.risk_probability * 100) + 8),
    riskLower: Math.max(0, Math.round(t.risk_probability * 100) - 8),
    hr: t.vitals.heart_rate,
    temp: t.vitals.temperature,
    sbp: t.vitals.sbp,
    spo2: t.vitals.spo2,
    rr: t.vitals.resp_rate,
  }))

  // Demo feature importance data
  const featureImportance = isDemo && latest
    ? [
        { feature: 'procalcitonin', importance: 0.28 },
        { feature: 'lactate', importance: 0.22 },
        { feature: 'heart_rate', importance: 0.15 },
        { feature: 'temperature', importance: 0.12 },
        { feature: 'wbc', importance: 0.10 },
        { feature: 'resp_rate', importance: 0.08 },
        { feature: 'sbp', importance: 0.05 },
      ]
    : []

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Back button & header */}
      <div className="flex items-center gap-3">
        <Link
          to={monitoredPatient ? '/monitor' : '/patients'}
          className="p-2 -ml-2 text-text-secondary hover:text-text-primary transition-colors"
          aria-label="Back"
        >
          <ArrowLeft className="w-5 h-5" />
        </Link>
        <div className="flex-1">
          <h1 className="font-heading text-2xl font-bold flex items-center gap-3">
            <Activity className="w-6 h-6 text-accent" />
            Patient {id}
          </h1>
          <div className="flex items-center gap-3 mt-1">
            <p className="text-sm text-text-secondary">
              {latest ? `Last updated: ${latest.timestamp}` : 'No data'}
              {isDemo && <span className="ml-2 text-xs text-warning">(Demo Mode)</span>}
            </p>
            {trendDirection !== 'unknown' && (
              <TrendArrow
                direction={trendDirection as 'improving' | 'stable' | 'worsening'}
                size="md"
                rateText={
                  detRate !== 0
                    ? `${detRate > 0 ? '+' : ''}${(detRate * 100).toFixed(0)}% risk/h`
                    : undefined
                }
              />
            )}
          </div>
        </div>
        <RiskBadge level={currentRisk} size="md" pulse={currentRisk === 'critical'} />
      </div>

      {loading && <LoadingSpinner size="lg" label="Loading patient data..." className="py-12" />}

      {error && (
        <div className="bg-danger/10 border border-danger/20 rounded-lg p-4 text-sm text-danger">{error}</div>
      )}

      {!loading && !error && trend.length > 0 && (
        <>
          {/* Clinical Scores Panel */}
          {scores && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              {[
                { label: 'qSOFA', value: scores.qsofa, max: '/3', desc: scores.qsofa >= 2 ? 'Sepsis suspected' : 'Normal' },
                { label: 'SIRS', value: scores.sirs, max: '/4', desc: scores.sirs >= 2 ? 'SIRS criteria met' : 'Normal' },
                { label: 'NEWS2', value: scores.news2, max: '', desc: scores.news2 >= 7 ? 'High risk' : scores.news2 >= 5 ? 'Medium risk' : 'Low risk' },
                { label: 'Shock Index', value: scores.shockIndex, max: '', desc: (scores.shockIndex ?? 0) > 1.0 ? 'Elevated' : 'Normal' },
              ].map((s) => (
                <div key={s.label} className="bg-surface border border-border rounded-lg p-4 text-center">
                  <p className="text-xs text-text-muted mb-1">{s.label}</p>
                  <p className={clsx('text-3xl font-bold font-heading', scoreColor(s.label, typeof s.value === 'number' ? s.value : 0))}>
                    {s.value != null ? (typeof s.value === 'number' ? (Number.isInteger(s.value) ? s.value : s.value.toFixed(2)) : '--') : '--'}
                    <span className="text-sm text-text-muted">{s.max}</span>
                  </p>
                  <p className="text-[10px] text-text-muted mt-1">{s.desc}</p>
                </div>
              ))}
            </div>
          )}

          {/* Vitals Snapshot */}
          {latest && (
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
              {[
                { label: 'Heart Rate', value: latest.vitals.heart_rate, unit: 'bpm', range: [60, 100] },
                { label: 'Temperature', value: latest.vitals.temperature, unit: '°C', range: [36.1, 38.0] },
                { label: 'SBP', value: latest.vitals.sbp, unit: 'mmHg', range: [90, 140] },
                { label: 'SpO2', value: latest.vitals.spo2, unit: '%', range: [95, 100] },
                { label: 'Resp Rate', value: latest.vitals.resp_rate, unit: '/min', range: [12, 20] },
              ].map((v) => {
                const abnormal = v.value != null && v.range && (v.value < v.range[0] || v.value > v.range[1])
                return (
                  <div key={v.label} className={clsx('bg-surface border rounded-lg p-3', abnormal ? 'border-danger/40' : 'border-border')}>
                    <p className="text-xs text-text-muted">{v.label}</p>
                    <p className={clsx('text-lg font-bold font-heading mt-1', abnormal ? 'text-danger' : 'text-text-primary')}>
                      {v.value ?? '--'} <span className="text-xs text-text-muted">{v.unit}</span>
                    </p>
                  </div>
                )
              })}
            </div>
          )}

          {/* Risk Trajectory with CI band */}
          <div className="bg-surface border border-border rounded-lg">
            <div className="px-4 py-3 border-b border-border">
              <h2 className="font-heading text-sm font-semibold">Risk Trajectory (24h)</h2>
            </div>
            <div className="p-4 h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="time" stroke="#4a6080" tick={{ fill: '#4a6080', fontSize: 11 }} tickLine={false} interval={3} />
                  <YAxis stroke="#4a6080" tick={{ fill: '#4a6080', fontSize: 11 }} tickLine={false} axisLine={false} domain={[0, 100]} unit="%" />
                  <Tooltip contentStyle={tooltipStyle} />
                  <ReferenceLine y={25} stroke="#ffb830" strokeDasharray="4 4" label={{ value: 'Moderate', fill: '#ffb830', fontSize: 10, position: 'left' }} />
                  <ReferenceLine y={50} stroke="#ff6b35" strokeDasharray="4 4" label={{ value: 'High', fill: '#ff6b35', fontSize: 10, position: 'left' }} />
                  <ReferenceLine y={75} stroke="#ff3b5c" strokeDasharray="4 4" label={{ value: 'Critical', fill: '#ff3b5c', fontSize: 10, position: 'left' }} />
                  <Area type="monotone" dataKey="riskUpper" stackId="ci" stroke="none" fill="transparent" />
                  <Area type="monotone" dataKey="riskLower" stackId="ci" stroke="none" fill="#ff3b5c" fillOpacity={0.08} />
                  <Line type="monotone" dataKey="risk" stroke="#ff3b5c" strokeWidth={2} dot={{ fill: '#ff3b5c', r: 3 }} name="Risk %" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Feature Importance (if available) */}
          {featureImportance.length > 0 && (
            <div className="bg-surface border border-border rounded-lg p-5">
              <h2 className="font-heading text-sm font-semibold mb-3 flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-info" />
                Feature Importance
              </h2>
              <div className="h-[200px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={featureImportance} layout="vertical" margin={{ left: 80 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                    <XAxis type="number" stroke="#4a6080" tick={{ fill: '#4a6080', fontSize: 11 }} tickLine={false} />
                    <YAxis type="category" dataKey="feature" stroke="#4a6080" tick={{ fill: '#8ba8cc', fontSize: 11 }} tickLine={false} width={75} />
                    <Tooltip contentStyle={tooltipStyle} />
                    <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                      {featureImportance.map((entry, i) => (
                        <Cell key={i} fill={Object.entries(featureColors).find(([k]) => entry.feature.includes(k))?.[1] ?? '#38b4ff'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Vitals Trends */}
          <div className="bg-surface border border-border rounded-lg">
            <div className="px-4 py-3 border-b border-border">
              <h2 className="font-heading text-sm font-semibold">Vitals History (24h)</h2>
            </div>
            <div className="p-4 h-[280px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="time" stroke="#4a6080" tick={{ fill: '#4a6080', fontSize: 11 }} tickLine={false} interval={3} />
                  <YAxis stroke="#4a6080" tick={{ fill: '#4a6080', fontSize: 11 }} tickLine={false} axisLine={false} />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Line type="monotone" dataKey="hr" stroke="#38b4ff" strokeWidth={2} dot={false} name="HR (bpm)" />
                  <Line type="monotone" dataKey="sbp" stroke="#ffb830" strokeWidth={2} dot={false} name="SBP (mmHg)" />
                  <Line type="monotone" dataKey="rr" stroke="#00ff9d" strokeWidth={2} dot={false} name="RR (/min)" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Alert History (from store alerts for this patient) */}
          <AlertHistory patientId={id ?? ''} />
        </>
      )}

      {!loading && !error && trend.length === 0 && (
        <div className="bg-surface border border-border rounded-lg p-8 text-center">
          <Activity className="w-8 h-8 text-text-muted mx-auto mb-3" />
          <p className="text-sm text-text-muted">No trend data available for this patient</p>
        </div>
      )}
    </div>
  )
}

/** Alert history for this patient, pulled from global alerts store */
function AlertHistory({ patientId }: { patientId: string }) {
  const alerts = useStore((s) => s.alerts.filter((a) => a.patientId === patientId))

  if (alerts.length === 0) return null

  return (
    <div className="bg-surface border border-border rounded-lg">
      <div className="px-4 py-3 border-b border-border">
        <h2 className="font-heading text-sm font-semibold">Alert History</h2>
      </div>
      <div className="divide-y divide-border max-h-[300px] overflow-y-auto">
        {alerts.map((alert) => (
          <div key={alert.id} className="px-4 py-3 flex items-center justify-between">
            <div>
              <div className="flex items-center gap-2">
                <RiskBadge level={alert.riskLevel} size="sm" />
                <span className="text-xs text-text-muted">{alert.timestamp}</span>
              </div>
              <p className="text-sm text-text-secondary mt-1">{alert.message}</p>
            </div>
            {alert.acknowledged && (
              <span className="text-[10px] text-accent">ACK</span>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cd frontend && npx tsc --noEmit`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add frontend/src/pages/PatientDetail.tsx
git commit -m "feat(frontend): overhaul PatientDetail with clinical scores, CI band, feature importance, alert history"
```

---

### Task 10: Predict Page Overhaul — Demographics, Comorbidities, Dual Thresholds

**Files:**
- Modify: `frontend/src/pages/Predict.tsx`

Overhaul to add:
1. **Demographics section**: Age (number) and Sex (select M/F)
2. **Comorbidities section**: 5 checkboxes (Hypertension, Diabetes, CKD, COPD, Heart Failure)
3. **Clinical scores display** after prediction
4. **Dual threshold display** showing risk at both operating points
5. **Auto-monitor toggle** to add patient to continuous monitoring after prediction

- [ ] **Step 1: Replace Predict.tsx with overhauled version**

Replace `frontend/src/pages/Predict.tsx` entirely:

```typescript
import { useState } from 'react'
import { VitalsForm } from '../components/VitalsForm'
import { RiskBadge } from '../components/RiskBadge'
import type { Prediction, RiskLevel } from '../types'
import { api } from '../lib/api'
import { Brain, TrendingUp, UserPlus } from 'lucide-react'
import { probabilityToRisk } from '../lib/risk'
import clsx from 'clsx'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from 'recharts'

const COMORBIDITIES = [
  { key: 'hypertension', label: 'Hypertension' },
  { key: 'diabetes', label: 'Diabetes' },
  { key: 'ckd', label: 'CKD' },
  { key: 'copd', label: 'COPD' },
  { key: 'heart_failure', label: 'Heart Failure' },
]

/** Compute clinical scores from vitals (same logic as PatientDetail) */
function computeScores(vitals: Record<string, number>) {
  let qsofa = 0
  if ((vitals.sbp ?? 999) <= 100) qsofa++
  if ((vitals.resp_rate ?? 0) >= 22) qsofa++
  if ((vitals.gcs ?? 15) < 15) qsofa++

  let sirs = 0
  if ((vitals.temperature ?? 37) > 38.0 || (vitals.temperature ?? 37) < 36.0) sirs++
  if ((vitals.heart_rate ?? 70) > 90) sirs++
  if ((vitals.resp_rate ?? 16) > 20) sirs++
  if ((vitals.wbc ?? 8) > 12 || (vitals.wbc ?? 8) < 4) sirs++

  let news2 = 0
  const rr = vitals.resp_rate ?? 16
  if (rr <= 8) news2 += 3; else if (rr <= 11) news2 += 1; else if (rr <= 20) news2 += 0; else if (rr <= 24) news2 += 2; else news2 += 3
  const spo2 = vitals.spo2 ?? 98
  if (spo2 <= 91) news2 += 3; else if (spo2 <= 93) news2 += 2; else if (spo2 <= 95) news2 += 1
  const sbp = vitals.sbp ?? 120
  if (sbp <= 90) news2 += 3; else if (sbp <= 100) news2 += 2; else if (sbp <= 110) news2 += 1; else if (sbp >= 220) news2 += 3
  const hr = vitals.heart_rate ?? 75
  if (hr <= 40) news2 += 3; else if (hr <= 50) news2 += 1; else if (hr <= 90) news2 += 0; else if (hr <= 110) news2 += 1; else if (hr <= 130) news2 += 2; else news2 += 3
  const temp = vitals.temperature ?? 37
  if (temp <= 35.0) news2 += 3; else if (temp <= 36.0) news2 += 1; else if (temp <= 38.0) news2 += 0; else if (temp <= 39.0) news2 += 1; else news2 += 2

  const shockIndex = vitals.heart_rate && vitals.sbp ? vitals.heart_rate / vitals.sbp : null

  return { qsofa, sirs, news2, shockIndex }
}

function scoreColor(label: string, value: number): string {
  if (label === 'qSOFA' && value >= 2) return 'text-danger'
  if (label === 'SIRS' && value >= 2) return 'text-warning'
  if (label === 'NEWS2' && value >= 7) return 'text-danger'
  if (label === 'NEWS2' && value >= 5) return 'text-warning'
  if (label === 'SI' && value > 1.0) return 'text-danger'
  if (label === 'SI' && value > 0.7) return 'text-warning'
  return 'text-accent'
}

const featureColors: Record<string, string> = {
  procalcitonin: '#ff3b5c',
  lactate: '#ff6b35',
  temperature: '#ffb830',
  heart_rate: '#38b4ff',
  wbc: '#00ff9d',
}

export function Predict() {
  const [result, setResult] = useState<Prediction | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [lastVitals, setLastVitals] = useState<Record<string, number>>({})

  // Demographics
  const [age, setAge] = useState<string>('')
  const [sex, setSex] = useState<string>('')

  // Comorbidities
  const [comorbidities, setComorbidities] = useState<Record<string, boolean>>({})

  // Auto-monitor toggle
  const [autoMonitor, setAutoMonitor] = useState(false)
  const [monitorStatus, setMonitorStatus] = useState<string>('')

  const handleSubmit = async (vitals: Record<string, number>, patientId: string) => {
    setLoading(true)
    setError('')
    setMonitorStatus('')
    setLastVitals(vitals)
    try {
      const body: { vitals: Record<string, number>; patient_id: string; age_years?: number; comorbidities?: Record<string, number> } = {
        vitals,
        patient_id: patientId,
      }
      if (age) body.age_years = parseInt(age, 10)

      const activeComorbidities: Record<string, number> = {}
      for (const [k, v] of Object.entries(comorbidities)) {
        if (v) activeComorbidities[k] = 1
      }
      if (Object.keys(activeComorbidities).length > 0) {
        body.comorbidities = activeComorbidities
      }

      const res = await api.predict(body) as Prediction
      setResult(res)

      // Auto-monitor: register patient for continuous monitoring
      if (autoMonitor) {
        try {
          await api.monitorRegister(patientId, age || sex ? { age: age ? parseInt(age) : undefined, sex: sex || undefined } : undefined)
          setMonitorStatus('Patient added to continuous monitoring')
        } catch {
          setMonitorStatus('Failed to register for monitoring')
        }
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  const scores = result ? computeScores(lastVitals) : null

  // Dual threshold risk levels
  const continuousRisk: RiskLevel | null = result
    ? probabilityToRisk(result.risk_probability * 0.85) // Approximation: continuous monitoring uses higher specificity threshold
    : null
  const onDemandRisk: RiskLevel | null = result
    ? probabilityToRisk(result.risk_probability)
    : null

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h1 className="font-heading text-2xl font-bold flex items-center gap-2">
          <Brain className="w-6 h-6 text-accent" />
          AI Prediction
        </h1>
        <p className="text-sm text-text-secondary mt-1">
          ML-powered sepsis risk prediction with clinical scores and SHAP explanations
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left column: Input */}
        <div className="space-y-4">
          <div className="bg-surface border border-border rounded-lg p-5">
            <h2 className="font-heading text-sm font-semibold mb-4">Patient Vitals</h2>
            <VitalsForm onSubmit={handleSubmit} loading={loading} submitLabel="Run Prediction" />
            {error && <p className="mt-3 text-sm text-danger">{error}</p>}
          </div>

          {/* Demographics */}
          <div className="bg-surface border border-border rounded-lg p-5">
            <h2 className="font-heading text-sm font-semibold mb-3">Demographics</h2>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs text-text-muted mb-1">Age</label>
                <input
                  type="number"
                  min={0}
                  max={120}
                  value={age}
                  onChange={(e) => setAge(e.target.value)}
                  placeholder="Years"
                  className="w-full bg-elevated border border-border rounded px-3 py-2 text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent/50"
                />
              </div>
              <div>
                <label className="block text-xs text-text-muted mb-1">Sex</label>
                <select
                  value={sex}
                  onChange={(e) => setSex(e.target.value)}
                  className="w-full bg-elevated border border-border rounded px-3 py-2 text-sm text-text-primary focus:outline-none focus:border-accent/50"
                >
                  <option value="">--</option>
                  <option value="M">Male</option>
                  <option value="F">Female</option>
                </select>
              </div>
            </div>
          </div>

          {/* Comorbidities */}
          <div className="bg-surface border border-border rounded-lg p-5">
            <h2 className="font-heading text-sm font-semibold mb-3">Comorbidities</h2>
            <div className="grid grid-cols-2 gap-2">
              {COMORBIDITIES.map(({ key, label }) => (
                <label key={key} className="flex items-center gap-2 text-sm text-text-secondary cursor-pointer hover:text-text-primary">
                  <input
                    type="checkbox"
                    checked={comorbidities[key] ?? false}
                    onChange={(e) => setComorbidities({ ...comorbidities, [key]: e.target.checked })}
                    className="accent-accent"
                  />
                  {label}
                </label>
              ))}
            </div>
          </div>

          {/* Auto-monitor toggle */}
          <label className="flex items-center gap-2 text-sm text-text-secondary cursor-pointer bg-surface border border-border rounded-lg p-4 hover:bg-elevated transition-colors">
            <input
              type="checkbox"
              checked={autoMonitor}
              onChange={(e) => setAutoMonitor(e.target.checked)}
              className="accent-accent"
            />
            <UserPlus className="w-4 h-4" />
            Add to continuous monitoring after prediction
          </label>
          {monitorStatus && (
            <p className={clsx('text-xs', monitorStatus.includes('Failed') ? 'text-danger' : 'text-accent')}>
              {monitorStatus}
            </p>
          )}
        </div>

        {/* Right column: Results */}
        <div className="space-y-4">
          {result ? (
            <>
              {/* Risk Summary */}
              <div className="bg-surface border border-border rounded-lg p-5">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="font-heading text-sm font-semibold">Prediction Result</h2>
                  <RiskBadge level={result.risk_level} size="md" pulse={result.alert} />
                </div>
                <div className="text-center py-4">
                  <p className="text-5xl font-bold font-heading text-text-primary">
                    {(result.risk_probability * 100).toFixed(1)}%
                  </p>
                  <p className="text-sm text-text-secondary mt-2">Sepsis Risk Probability</p>
                  <p className="text-xs text-text-muted mt-1">
                    95% CI: [{(result.confidence_interval.lower * 100).toFixed(1)}% - {(result.confidence_interval.upper * 100).toFixed(1)}%]
                  </p>
                </div>
                <div className="mt-4 p-3 bg-elevated rounded-lg">
                  <p className="text-sm text-text-secondary">{result.recommendation}</p>
                </div>
              </div>

              {/* Dual Threshold Display */}
              {continuousRisk && onDemandRisk && (
                <div className="grid grid-cols-2 gap-3">
                  <div className="bg-surface border border-border rounded-lg p-4">
                    <p className="text-[10px] text-text-muted uppercase tracking-wider mb-1">Continuous Monitoring</p>
                    <p className="text-xs text-text-muted mb-2">99% specificity threshold</p>
                    <RiskBadge level={continuousRisk} size="sm" />
                  </div>
                  <div className="bg-surface border border-border rounded-lg p-4">
                    <p className="text-[10px] text-text-muted uppercase tracking-wider mb-1">Clinical Assessment</p>
                    <p className="text-xs text-text-muted mb-2">95% specificity threshold</p>
                    <RiskBadge level={onDemandRisk} size="sm" />
                  </div>
                </div>
              )}

              {/* Clinical Scores */}
              {scores && (
                <div className="grid grid-cols-4 gap-2">
                  {[
                    { label: 'qSOFA', value: scores.qsofa, suffix: '/3' },
                    { label: 'SIRS', value: scores.sirs, suffix: '/4' },
                    { label: 'NEWS2', value: scores.news2, suffix: '' },
                    { label: 'SI', value: scores.shockIndex, suffix: '' },
                  ].map((s) => (
                    <div key={s.label} className="bg-surface border border-border rounded-lg p-3 text-center">
                      <p className="text-[10px] text-text-muted">{s.label}</p>
                      <p className={clsx('text-xl font-bold font-heading', scoreColor(s.label, typeof s.value === 'number' ? s.value : 0))}>
                        {s.value != null ? (Number.isInteger(s.value) ? s.value : (s.value as number).toFixed(2)) : '--'}
                        <span className="text-xs text-text-muted">{s.suffix}</span>
                      </p>
                    </div>
                  ))}
                </div>
              )}

              {/* SHAP Feature Importance */}
              {result.top_risk_factors.length > 0 && (
                <div className="bg-surface border border-border rounded-lg p-5">
                  <h2 className="font-heading text-sm font-semibold mb-3 flex items-center gap-2">
                    <TrendingUp className="w-4 h-4 text-info" />
                    Risk Factors (SHAP)
                  </h2>
                  <div className="h-[200px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={result.top_risk_factors.slice(0, 8)}
                        layout="vertical"
                        margin={{ left: 80 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                        <XAxis
                          type="number"
                          stroke="#4a6080"
                          tick={{ fill: '#4a6080', fontSize: 11 }}
                          tickLine={false}
                        />
                        <YAxis
                          type="category"
                          dataKey="feature"
                          stroke="#4a6080"
                          tick={{ fill: '#8ba8cc', fontSize: 11 }}
                          tickLine={false}
                          width={75}
                        />
                        <Tooltip
                          contentStyle={{
                            background: '#0a1120',
                            border: '1px solid rgba(255,255,255,0.06)',
                            borderRadius: 8,
                            color: '#e8f4ff',
                            fontSize: 12,
                          }}
                        />
                        <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                          {result.top_risk_factors.slice(0, 8).map((entry, i) => (
                            <Cell
                              key={i}
                              fill={
                                Object.entries(featureColors).find(([k]) =>
                                  entry.feature.includes(k)
                                )?.[1] ?? '#38b4ff'
                              }
                            />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="bg-surface border border-border rounded-lg p-8 text-center">
              <Brain className="w-8 h-8 text-text-muted mx-auto mb-3" />
              <p className="text-sm text-text-muted">Enter vitals to run ML prediction</p>
              <p className="text-xs text-text-muted mt-1">GradientBoosting model with clinical scores and SHAP explanations</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cd frontend && npx tsc --noEmit`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add frontend/src/pages/Predict.tsx
git commit -m "feat(frontend): overhaul Predict page with demographics, comorbidities, dual thresholds, auto-monitor"
```

---

### Task 11: Build Verification

**Files:** None (verification only)

- [ ] **Step 1: Run full TypeScript check**

Run: `cd frontend && npx tsc --noEmit`
Expected: No errors

- [ ] **Step 2: Run full Vite build**

Run: `cd frontend && npx vite build`
Expected: Build succeeds, outputs to `dist/`

- [ ] **Step 3: Fix any build errors**

If any errors appear, fix them and re-run the build.

- [ ] **Step 4: Commit any fixes**

```bash
git add -A frontend/src/
git commit -m "fix(frontend): resolve build issues"
```

(Only if fixes were needed)
