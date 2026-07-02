import { useEffect, useMemo, useState } from 'react'
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
import { useTranslation } from 'react-i18next'
import { BundlePanel } from '../components/BundlePanel'
import type { BundleClient } from '../components/BundlePanel'

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
  const { t } = useTranslation()
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

  const bundleClient: BundleClient = useMemo(() => ({
    getForPatient: (pid: string) => api.bundleGetForPatient(pid),
    start: (pid: string) => api.bundleStart(pid),
    completeTask: (bid: string, key: string, done: boolean) => api.bundleCompleteTask(bid, key, done),
    cancel: (bid: string) => api.bundleCancel(bid),
  }), [])

  const [forecast, setForecast] = useState<{
    trend_per_hour: number
    projected_risk_1h: number
    hours_to_critical: number | null
    lead_time_band: { low_hours: number; high_hours: number } | null
    horizon_label: string
    confidence: string
  } | null>(null)

  useEffect(() => {
    if (!id || isDemo) return
    api.patientForecast(id)
      .then(setForecast)
      .catch(() => {})
  }, [id])

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
            {t('patientDetail.title', { id })}
          </h1>
          <div className="flex items-center gap-3 mt-1">
            <p className="text-sm text-text-secondary">
              {latest
                ? t('patientDetail.lastUpdated', { timestamp: latest.timestamp })
                : t('common.noData')}
              {isDemo && <span className="ml-2 text-xs text-warning">{t('common.demoMode')}</span>}
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

      {loading && <LoadingSpinner size="lg" label={t('patientDetail.loadingPatient')} className="py-12" />}

      {error && (
        <div className="bg-danger/10 border border-danger/20 rounded-lg p-4 text-sm text-danger">{error}</div>
      )}

      {!loading && !error && trend.length > 0 && (
        <>
          {/* Clinical Scores Panel */}
          {scores && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              {[
                { label: t('scores.qsofa'), key: 'qSOFA', value: scores.qsofa, max: '/3', desc: scores.qsofa >= 2 ? t('scores.sepsisSupsp') : t('scores.normal') },
                { label: t('scores.sirs'), key: 'SIRS', value: scores.sirs, max: '/4', desc: scores.sirs >= 2 ? t('scores.criteriaMet') : t('scores.normal') },
                { label: t('scores.news2'), key: 'NEWS2', value: scores.news2, max: '', desc: scores.news2 >= 7 ? t('scores.highRisk') : scores.news2 >= 5 ? t('scores.mediumRisk') : t('scores.lowRisk') },
                { label: t('scores.si'), key: 'Shock Index', value: scores.shockIndex, max: '', desc: (scores.shockIndex ?? 0) > 1.0 ? t('scores.elevated') : t('scores.normal') },
              ].map((s) => (
                <div key={s.key} className="bg-surface border border-border rounded-lg p-4 text-center">
                  <p className="text-xs text-text-muted mb-1">{s.label}</p>
                  <p className={clsx('text-3xl font-bold font-heading', scoreColor(s.key, typeof s.value === 'number' ? s.value : 0))}>
                    {s.value != null ? (typeof s.value === 'number' ? (Number.isInteger(s.value) ? s.value : s.value.toFixed(2)) : '--') : '--'}
                    <span className="text-sm text-text-muted">{s.max}</span>
                  </p>
                  <p className="text-[10px] text-text-muted mt-1">{s.desc}</p>
                </div>
              ))}
            </div>
          )}

          {/* Hour-1 Bundle Tracker */}
          <BundlePanel patientId={id!} client={bundleClient} />

          {/* Deterioration Forecast */}
          {forecast && (
            <div className="bg-surface border border-border rounded-lg p-4">
              <h3 className="font-heading text-sm font-semibold mb-3 flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-info" />
                {t('patientDetail.forecast', 'Deterioration Forecast')}
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <div className="text-center">
                  <p className="text-[10px] uppercase tracking-wider text-text-muted">Trend</p>
                  <p className={clsx(
                    'text-lg font-bold font-heading',
                    forecast.trend_per_hour > 0.02 ? 'text-danger' : forecast.trend_per_hour < -0.02 ? 'text-accent' : 'text-text-primary',
                  )}>
                    {forecast.trend_per_hour > 0 ? '+' : ''}{(forecast.trend_per_hour * 100).toFixed(1)}%/h
                  </p>
                </div>
                <div className="text-center">
                  <p className="text-[10px] uppercase tracking-wider text-text-muted">1h Projected</p>
                  <p className="text-lg font-bold font-heading text-text-primary">
                    {(forecast.projected_risk_1h * 100).toFixed(0)}%
                  </p>
                </div>
                <div className="text-center">
                  <p className="text-[10px] uppercase tracking-wider text-text-muted">Time to Critical</p>
                  <p className="text-lg font-bold font-heading text-text-primary">
                    {forecast.hours_to_critical != null
                      ? forecast.hours_to_critical === 0
                        ? 'NOW'
                        : `~${forecast.hours_to_critical.toFixed(1)}h`
                      : '—'}
                  </p>
                </div>
                <div className="text-center">
                  <p className="text-[10px] uppercase tracking-wider text-text-muted">Confidence</p>
                  <p className={clsx(
                    'text-lg font-bold font-heading',
                    forecast.confidence === 'high' ? 'text-accent' : forecast.confidence === 'low' ? 'text-warning' : 'text-text-primary',
                  )}>
                    {forecast.confidence}
                  </p>
                </div>
              </div>
              <p className="text-xs text-text-secondary mt-3 text-center">{forecast.horizon_label}</p>
              {forecast.lead_time_band && (
                <p className="text-[10px] text-text-muted text-center mt-1">
                  Lead time band: {forecast.lead_time_band.low_hours.toFixed(1)}–{forecast.lead_time_band.high_hours.toFixed(1)}h
                </p>
              )}
            </div>
          )}

          {/* Vitals Snapshot */}
          {latest && (
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
              {[
                { label: t('vitals.heartRateFull'), value: latest.vitals.heart_rate, unit: t('vitals.units.bpm'), range: [60, 100] as [number, number] },
                { label: t('vitals.temperatureFull'), value: latest.vitals.temperature, unit: t('vitals.units.celsius'), range: [36.1, 38.0] as [number, number] },
                { label: t('vitals.sbp'), value: latest.vitals.sbp, unit: t('vitals.units.mmHg'), range: [90, 140] as [number, number] },
                { label: t('vitals.spo2'), value: latest.vitals.spo2, unit: t('vitals.units.percent'), range: [95, 100] as [number, number] },
                { label: t('vitals.respRateFull'), value: latest.vitals.resp_rate, unit: t('vitals.units.perMin'), range: [12, 20] as [number, number] },
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
              <h2 className="font-heading text-sm font-semibold">{t('patientDetail.riskTrajectory')}</h2>
            </div>
            <div className="p-4 h-[300px]">
              <div role="img" aria-label={t('patientDetail.riskTrajectoryLabel', { id })}>
              <ResponsiveContainer width="100%" height={268}>
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
          </div>

          {/* Feature Importance (if available) */}
          {featureImportance.length > 0 && (
            <div className="bg-surface border border-border rounded-lg p-5">
              <h2 className="font-heading text-sm font-semibold mb-3 flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-info" />
                {t('patientDetail.featureImportance')}
              </h2>
              <div className="h-[200px]" role="img" aria-label={t('patientDetail.featureImportanceLabel')}>
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
              <h2 className="font-heading text-sm font-semibold">{t('patientDetail.vitalsHistory')}</h2>
            </div>
            <div className="p-4 h-[280px]">
              <div role="img" aria-label={t('patientDetail.vitalsHistoryLabel', { id })}>
              <ResponsiveContainer width="100%" height={248}>
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
          </div>

          {/* Alert History */}
          <AlertHistory patientId={id ?? ''} />
        </>
      )}

      {!loading && !error && trend.length === 0 && (
        <div className="bg-surface border border-border rounded-lg p-8 text-center">
          <Activity className="w-8 h-8 text-text-muted mx-auto mb-3" />
          <p className="text-sm text-text-muted">{t('patientDetail.noTrendData')}</p>
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
