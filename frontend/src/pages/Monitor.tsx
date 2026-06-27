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
      alert_state: 'normal' as const,
      deterioration_rate: 0,
      window_hours: 0,
      risk_history: history,
    }
  })
}

/** Stale threshold: 30 minutes without telemetry = signal lost */
const STALE_THRESHOLD_SECONDS = 30 * 60

function timeAgo(unixSeconds: number): string {
  const diff = Date.now() / 1000 - unixSeconds
  if (diff < 60) return 'just now'
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`
  return `${Math.floor(diff / 86400)}d ago`
}

function isStale(patient: MonitoredPatient): boolean {
  const lastData = patient.last_vitals_time || patient.last_prediction_time
  if (!lastData) return true
  return (Date.now() / 1000 - lastData) > STALE_THRESHOLD_SECONDS
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
            aria-label="Sort by risk level"
            aria-pressed={sortByRisk}
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
            aria-label="Refresh monitor"
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
          {patients.map((patient) => {
            const stale = isStale(patient)
            return (
            <Link
              key={patient.patient_id}
              to={`/patients/${patient.patient_id}`}
              className={clsx(
                'bg-surface border-2 rounded-lg p-4 transition-all hover:bg-elevated block',
                stale ? 'border-border opacity-60' : RISK_BORDER[patient.risk_level],
                !stale && patient.risk_level === 'critical' && 'animate-pulse-critical',
              )}
            >
              {/* Stale banner */}
              {stale && (
                <div className="bg-warning/10 border border-warning/30 rounded px-2 py-1 mb-3 text-[10px] text-warning font-medium text-center">
                  Telemetry Lost — Data Stale
                </div>
              )}

              {/* Header row */}
              <div className="flex items-center justify-between mb-3">
                <span className={clsx('font-heading font-semibold text-sm', stale && 'text-text-muted')}>{patient.patient_id}</span>
                {stale
                  ? <span className="text-[10px] px-2 py-0.5 rounded-full bg-elevated text-text-muted border border-border">Stale</span>
                  : <RiskBadge level={patient.risk_level} size="sm" pulse={patient.risk_level === 'critical'} />
                }
              </div>

              {/* Risk probability + trend */}
              <div className="flex items-center justify-between mb-3">
                <span className={clsx('text-2xl font-bold font-heading', stale && 'text-text-muted')}>
                  {stale ? '?' : `${(patient.risk_probability * 100).toFixed(0)}%`}
                </span>
                {!stale && (
                  <TrendArrow
                    direction={patient.trend_direction}
                    rateText={
                      patient.deterioration_rate
                        ? `${patient.deterioration_rate > 0 ? '+' : ''}${(patient.deterioration_rate * 100).toFixed(0)}%/h`
                        : undefined
                    }
                  />
                )}
              </div>

              {/* 24h Sparkline */}
              {patient.risk_history && patient.risk_history.length > 1 && (
                <div className="h-[40px] mb-3">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={patient.risk_history}>
                      <Line
                        type="monotone"
                        dataKey="risk_probability"
                        stroke={stale ? '#666' : (
                          patient.risk_level === 'critical' ? '#ff3b5c' :
                          patient.risk_level === 'high' ? '#ff6b35' :
                          patient.risk_level === 'moderate' ? '#ffb830' : '#00ff9d'
                        )}
                        strokeWidth={1.5}
                        dot={false}
                        strokeDasharray={stale ? '4 4' : undefined}
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
                      <div className={clsx('font-medium', stale ? 'text-text-muted' : isAbnormal(key, value) ? 'text-danger' : 'text-text-primary')}>
                        {key === 'temperature' ? value.toFixed(1) : Math.round(value)}
                        <span className="text-text-muted ml-0.5">{units[key]}</span>
                      </div>
                    </div>
                  )
                })}
              </div>

              {/* Last updated */}
              <div className={clsx('text-[10px] text-right', stale ? 'text-warning font-medium' : 'text-text-muted')}>
                {timeAgo(patient.last_vitals_time || patient.last_prediction_time)}
              </div>
            </Link>
            )
          })}
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
