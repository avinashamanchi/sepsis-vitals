import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { ArrowLeft, Activity } from 'lucide-react'
import { api, isDemo } from '../lib/api'
import { RiskBadge } from '../components/RiskBadge'
import { LoadingSpinner } from '../components/LoadingSpinner'
import type { RiskLevel } from '../types'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts'

interface TrendPoint {
  timestamp: string
  risk_probability: number
  vitals: Record<string, number>
}

/** Generate deterministic demo data from a patient ID. */
function makeDemoTrend(id: string): TrendPoint[] {
  // Use char codes as a simple seed
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

export function PatientDetail() {
  const { id } = useParams<{ id: string }>()
  const [trend, setTrend] = useState<TrendPoint[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!id) return
    if (isDemo) {
      setTrend(makeDemoTrend(id))
      setLoading(false)
      return
    }
    api.patientTrend(id)
      .then((data) => {
        setTrend(data.trend ?? [])
      })
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : 'Failed to load patient data')
      })
      .finally(() => setLoading(false))
  }, [id])

  const latest = trend.length > 0 ? trend[trend.length - 1] : null
  const currentRisk: RiskLevel = latest ? riskFromProb(latest.risk_probability) : 'low'

  const chartData = trend.map((t) => ({
    time: t.timestamp,
    risk: Math.round(t.risk_probability * 100),
    hr: t.vitals.heart_rate,
    temp: t.vitals.temperature,
    sbp: t.vitals.sbp,
  }))

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Back button & header */}
      <div className="flex items-center gap-3">
        <Link
          to="/patients"
          className="p-2 -ml-2 text-text-secondary hover:text-text-primary transition-colors"
          aria-label="Back to patients"
        >
          <ArrowLeft className="w-5 h-5" />
        </Link>
        <div className="flex-1">
          <h1 className="font-heading text-2xl font-bold flex items-center gap-3">
            <Activity className="w-6 h-6 text-accent" />
            Patient {id}
          </h1>
          <p className="text-sm text-text-secondary mt-1">
            {latest ? `Last updated: ${latest.timestamp}` : 'No data'}
            {isDemo && <span className="ml-2 text-xs text-warning">(Demo Mode)</span>}
          </p>
        </div>
        <RiskBadge level={currentRisk} size="md" pulse={currentRisk === 'critical'} />
      </div>

      {loading && (
        <LoadingSpinner size="lg" label="Loading patient data..." className="py-12" />
      )}

      {error && (
        <div className="bg-danger/10 border border-danger/20 rounded-lg p-4 text-sm text-danger">
          {error}
        </div>
      )}

      {!loading && !error && trend.length > 0 && (
        <>
          {/* Vitals Snapshot */}
          {latest && (
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
              {[
                { label: 'Heart Rate', value: `${latest.vitals.heart_rate ?? '--'} bpm` },
                { label: 'Temperature', value: `${latest.vitals.temperature ?? '--'}°C` },
                { label: 'SBP', value: `${latest.vitals.sbp ?? '--'} mmHg` },
                { label: 'SpO2', value: `${latest.vitals.spo2 ?? '--'}%` },
                { label: 'Resp Rate', value: `${latest.vitals.resp_rate ?? '--'}/min` },
              ].map((v) => (
                <div key={v.label} className="bg-surface border border-border rounded-lg p-3">
                  <p className="text-xs text-text-muted">{v.label}</p>
                  <p className="text-lg font-bold font-heading mt-1">{v.value}</p>
                </div>
              ))}
            </div>
          )}

          {/* Risk Trajectory */}
          <div className="bg-surface border border-border rounded-lg">
            <div className="px-4 py-3 border-b border-border">
              <h2 className="font-heading text-sm font-semibold">Risk Trajectory (24h)</h2>
            </div>
            <div className="p-4 h-[280px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="time" stroke="#4a6080" tick={{ fill: '#4a6080', fontSize: 11 }} tickLine={false} interval={3} />
                  <YAxis stroke="#4a6080" tick={{ fill: '#4a6080', fontSize: 11 }} tickLine={false} axisLine={false} domain={[0, 100]} unit="%" />
                  <Tooltip
                    contentStyle={{
                      background: '#0a1120',
                      border: '1px solid rgba(255,255,255,0.06)',
                      borderRadius: 8,
                      color: '#e8f4ff',
                      fontSize: 12,
                    }}
                  />
                  <Line type="monotone" dataKey="risk" stroke="#ff3b5c" strokeWidth={2} dot={{ fill: '#ff3b5c', r: 3 }} name="Risk %" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

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
                  <Tooltip
                    contentStyle={{
                      background: '#0a1120',
                      border: '1px solid rgba(255,255,255,0.06)',
                      borderRadius: 8,
                      color: '#e8f4ff',
                      fontSize: 12,
                    }}
                  />
                  <Line type="monotone" dataKey="hr" stroke="#38b4ff" strokeWidth={2} dot={false} name="HR (bpm)" />
                  <Line type="monotone" dataKey="sbp" stroke="#ffb830" strokeWidth={2} dot={false} name="SBP (mmHg)" />
                  <Line type="monotone" dataKey="temp" stroke="#00ff9d" strokeWidth={2} dot={false} name="Temp (°C)" yAxisId="temp" hide />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
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
