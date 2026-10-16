import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { StatCard } from '../components/StatCard'
import { BarChart3 } from 'lucide-react'
import { api, isDemo } from '../lib/api'
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, PieChart, Pie, Cell,
} from 'recharts'

const DEMO_WEEKLY = [
  { day: 'Mon', predictions: 142, alerts: 8, dismissed: 3 },
  { day: 'Tue', predictions: 168, alerts: 12, dismissed: 5 },
  { day: 'Wed', predictions: 155, alerts: 6, dismissed: 2 },
  { day: 'Thu', predictions: 189, alerts: 15, dismissed: 7 },
  { day: 'Fri', predictions: 201, alerts: 11, dismissed: 4 },
  { day: 'Sat', predictions: 134, alerts: 9, dismissed: 3 },
  { day: 'Sun', predictions: 121, alerts: 5, dismissed: 1 },
]

const DEMO_RISK_DIST = [
  { name: 'Low', value: 58, color: '#00ff9d' },
  { name: 'Moderate', value: 24, color: '#ffb830' },
  { name: 'High', value: 12, color: '#ff6b35' },
  { name: 'Critical', value: 6, color: '#ff3b5c' },
]

const CHART_TOOLTIP = {
  contentStyle: {
    background: '#0a1120',
    border: '1px solid rgba(255,255,255,0.06)',
    borderRadius: 8,
    color: '#e8f4ff',
    fontSize: 12,
  },
}

const RISK_COLORS: Record<string, string> = {
  low: '#00ff9d', moderate: '#ffb830', high: '#ff6b35', critical: '#ff3b5c',
}

type WeeklyPoint = { day: string; predictions: number; alerts: number; dismissed?: number }
type RiskSlice = { name: string; value: number; color: string }

export function Analytics() {
  const { t } = useTranslation()
  const [weeklyData, setWeeklyData] = useState<WeeklyPoint[]>(isDemo ? DEMO_WEEKLY : [])
  const [riskDist, setRiskDist] = useState<RiskSlice[]>(isDemo ? DEMO_RISK_DIST : [])
  // Illustrative figures only in demo mode; live mode shows observed counts or '—'.
  const [stats, setStats] = useState(isDemo
    ? { totalPredictions: '1,110', alertsGenerated: '66', alertRate: '6.0%', reviewedFlags: '25', dataSource: 'Synthetic' }
    : { totalPredictions: '—', alertsGenerated: '—', alertRate: '—', reviewedFlags: '—', dataSource: 'Live counts' })

  useEffect(() => {
    if (isDemo) return
    // Observed daily counts from the backend: no extrapolation.
    api.weeklyTrends(7)
      .then((days) => {
        const total = days.reduce((n, d) => n + d.predictions, 0)
        const flagged = days.reduce((n, d) => n + d.alerts, 0)
        setWeeklyData(days.map((d) => ({ day: d.date ?? '—', predictions: d.predictions, alerts: d.alerts })))
        setStats((s) => ({
          ...s,
          totalPredictions: total.toLocaleString(),
          alertsGenerated: String(flagged),
          alertRate: total > 0 ? `${((flagged / total) * 100).toFixed(1)}%` : '—',
        }))
      })
      .catch((err: unknown) => console.error('Failed to load weekly trends:', err))
    api.riskDistribution(24)
      .then((rows) => setRiskDist(rows.map((r) => ({
        name: r.risk_level.charAt(0).toUpperCase() + r.risk_level.slice(1),
        value: r.percentage,
        color: RISK_COLORS[r.risk_level] ?? '#4a6080',
      }))))
      .catch((err: unknown) => console.error('Failed to load risk distribution:', err))
  }, [])
  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h1 className="font-heading text-2xl font-bold flex items-center gap-2">
          <BarChart3 className="w-6 h-6 text-info" />
          {t('analytics.title')}
        </h1>
        <p className="text-sm text-text-secondary mt-1">
          Evaluation volume and alert burden
          {isDemo && <span className="ml-2 text-xs text-warning">Synthetic scenario — not observed performance</span>}
        </p>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="Model outputs" value={stats.totalPredictions} sublabel={isDemo ? 'Evaluation window' : 'Last 7 days'} color="info" />
        <StatCard label="Review flags" value={stats.alertsGenerated} sublabel={`${stats.alertRate} flag rate`} color="warning" />
        <StatCard label="Flags reviewed" value={stats.reviewedFlags} sublabel="No outcome label implied" color="accent" />
        <StatCard label="Data source" value={stats.dataSource} sublabel={isDemo ? 'Interface demo' : 'Operational counts'} color="default" />
      </div>

      {!isDemo && (
        <div className="rounded-lg border border-info/20 bg-info/6 p-4 text-xs leading-5 text-text-secondary">
          Outcome metrics remain hidden until a site has a prespecified reference standard and
          adjudicated labels. Counting flags is not the same as measuring true positives.
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Predictions & Alerts */}
        <div className="bg-surface border border-border rounded-lg">
          <div className="px-4 py-3 border-b border-border">
            <h2 className="font-heading text-sm font-semibold">Outputs and review flags</h2>
          </div>
          <div className="p-4 h-[280px]">
            <div role="img" aria-label={t('analytics.predictionsVsAlertsLabel')}>
              <ResponsiveContainer width="100%" height={248}>
                <BarChart data={weeklyData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="day" stroke="#4a6080" tick={{ fill: '#4a6080', fontSize: 11 }} tickLine={false} />
                  <YAxis stroke="#4a6080" tick={{ fill: '#4a6080', fontSize: 11 }} tickLine={false} axisLine={false} />
                  <Tooltip {...CHART_TOOLTIP} />
                  <Bar dataKey="predictions" fill="#38b4ff" radius={[4, 4, 0, 0]} opacity={0.7} />
                  <Bar dataKey="alerts" fill="#ff3b5c" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Risk Distribution */}
        <div className="bg-surface border border-border rounded-lg">
          <div className="px-4 py-3 border-b border-border">
            <h2 className="font-heading text-sm font-semibold">{isDemo ? 'Synthetic output distribution' : 'Output distribution (last 24 h)'}</h2>
          </div>
          <div className="p-4 h-[280px] flex items-center">
            <div className="w-1/2 h-full" role="img" aria-label={t('analytics.riskDistributionLabel')}>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie data={riskDist} cx="50%" cy="50%" innerRadius={50} outerRadius={80} dataKey="value" paddingAngle={3}>
                    {riskDist.map((entry, i) => (
                      <Cell key={i} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip {...CHART_TOOLTIP} />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="w-1/2 space-y-3">
              {riskDist.map((d) => (
                <div key={d.name} className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-full" style={{ background: d.color }} />
                  <span className="text-sm text-text-secondary flex-1">{d.name}</span>
                  <span className="text-sm font-medium">{d.value}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Alert Fatigue Trend */}
      <div className="bg-surface border border-border rounded-lg">
        <div className="px-4 py-3 border-b border-border">
          <h2 className="font-heading text-sm font-semibold">Review-burden monitor</h2>
        </div>
        <div className="p-4 h-[240px]">
          <div role="img" aria-label={t('analytics.alertFatigueLabel')}>
            <ResponsiveContainer width="100%" height={208}>
              <LineChart data={weeklyData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis dataKey="day" stroke="#4a6080" tick={{ fill: '#4a6080', fontSize: 11 }} tickLine={false} />
                <YAxis stroke="#4a6080" tick={{ fill: '#4a6080', fontSize: 11 }} tickLine={false} axisLine={false} />
                <Tooltip {...CHART_TOOLTIP} />
                <Line type="monotone" dataKey="alerts" stroke="#ff3b5c" strokeWidth={2} dot={{ fill: '#ff3b5c', r: 4 }} />
                <Line type="monotone" dataKey="dismissed" stroke="#ffb830" strokeWidth={2} dot={{ fill: '#ffb830', r: 4 }} strokeDasharray="5 5" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  )
}
