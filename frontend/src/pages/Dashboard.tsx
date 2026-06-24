import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { StatCard } from '../components/StatCard'
import { AlertFeed } from '../components/AlertFeed'
import { RiskBadge } from '../components/RiskBadge'
import { useStore } from '../stores/useStore'
import { api, isDemo } from '../lib/api'
import { Activity, Brain, Users } from 'lucide-react'
import type { RiskLevel } from '../types'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts'

const DEMO_PATIENTS = [
  { id: 'P-1042', temp: 38.9, hr: 118, rr: 26, sbp: 88, spo2: 91, lac: 4.2, risk: 'critical' as RiskLevel, time: '2m ago' },
  { id: 'P-0891', temp: 38.4, hr: 105, rr: 22, sbp: 95, spo2: 94, lac: 2.8, risk: 'high' as RiskLevel, time: '5m ago' },
  { id: 'P-0756', temp: 37.8, hr: 92, rr: 20, sbp: 110, spo2: 96, lac: 1.5, risk: 'moderate' as RiskLevel, time: '8m ago' },
  { id: 'P-0623', temp: 37.0, hr: 78, rr: 16, sbp: 125, spo2: 98, lac: 0.9, risk: 'low' as RiskLevel, time: '12m ago' },
  { id: 'P-0512', temp: 36.8, hr: 72, rr: 14, sbp: 130, spo2: 99, lac: 0.7, risk: 'low' as RiskLevel, time: '15m ago' },
]

// Deterministic demo trend (not random on every render)
const DEMO_TREND = Array.from({ length: 24 }, (_, i) => ({
  hour: `${i}:00`,
  alerts: [0, 1, 0, 0, 1, 2, 1, 0, 0, 1, 3, 2, 1, 1, 0, 2, 4, 3, 1, 0, 0, 1, 0, 0][i],
  predictions: [3, 5, 4, 6, 8, 12, 14, 18, 22, 19, 16, 14, 12, 15, 18, 20, 24, 21, 17, 13, 9, 7, 5, 4][i],
}))

export function Dashboard() {
  const navigate = useNavigate()
  const alerts = useStore((s) => s.alerts)
  const patients = useStore((s) => s.patients)
  const wsConnected = useStore((s) => s.wsConnected)
  const activeAlerts = alerts.filter((a) => !a.dismissed).length

  const [stats, setStats] = useState({
    patientCount: patients.length || 12,
    predictionsToday: 147,
    modelAuroc: '0.92',
    modelName: 'GradientBoosting',
  })

  useEffect(() => {
    if (isDemo) return
    // Fetch real model info
    api.modelInfo()
      .then((info) => {
        const auroc = info.metrics?.val_auroc ?? info.metrics?.test_auroc
        setStats((s) => ({
          ...s,
          modelAuroc: auroc ? auroc.toFixed(2) : s.modelAuroc,
          modelName: info.model_name ?? s.modelName,
        }))
      })
      .catch((err: unknown) => console.error('Failed to load model info:', err))
    // Fetch real dashboard stats
    api.dashboardStats()
      .then((data) => {
        setStats((s) => ({
          ...s,
          patientCount: data.patient_count,
          predictionsToday: data.predictions_today,
        }))
      })
      .catch((err: unknown) => console.error('Failed to load dashboard stats:', err))
  }, [])

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div>
        <h1 className="font-heading text-2xl font-bold">Nurse Console</h1>
        <p className="text-sm text-text-secondary mt-1">
          Real-time patient monitoring
          {isDemo && <span className="ml-2 text-xs text-warning">(Demo Mode)</span>}
        </p>
      </div>

      {/* Stats Row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label="Active Patients"
          value={patients.length > 0 ? patients.length : stats.patientCount}
          sublabel="Being monitored"
          color="accent"
        />
        <StatCard
          label="Active Alerts"
          value={activeAlerts}
          sublabel="Require attention"
          color={activeAlerts > 0 ? 'danger' : 'default'}
        />
        <StatCard
          label="Predictions Today"
          value={stats.predictionsToday}
          sublabel="ML inferences"
          color="info"
        />
        <StatCard
          label="Model AUROC"
          value={stats.modelAuroc}
          sublabel={stats.modelName}
          color="accent"
        />
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Alert Feed */}
        <div className="lg:col-span-1">
          <div className="bg-surface border border-border rounded-lg">
            <div className="px-4 py-3 border-b border-border flex items-center justify-between">
              <h2 className="font-heading text-sm font-semibold flex items-center gap-2">
                <Activity className="w-4 h-4 text-danger" />
                Live Alerts
              </h2>
              <span className="text-xs text-text-muted">{wsConnected ? 'Connected' : 'Offline'}</span>
            </div>
            <div className="p-3">
              <AlertFeed limit={6} />
            </div>
          </div>
        </div>

        {/* Trend Chart */}
        <div className="lg:col-span-2">
          <div className="bg-surface border border-border rounded-lg">
            <div className="px-4 py-3 border-b border-border">
              <h2 className="font-heading text-sm font-semibold flex items-center gap-2">
                <Brain className="w-4 h-4 text-info" />
                24h Alert Trend
              </h2>
            </div>
            <div className="p-4 h-[280px]">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={DEMO_TREND}>
                  <defs>
                    <linearGradient id="alertGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#ff3b5c" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#ff3b5c" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="predGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#00ff9d" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#00ff9d" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis
                    dataKey="hour"
                    stroke="#4a6080"
                    tick={{ fill: '#4a6080', fontSize: 11 }}
                    tickLine={false}
                    interval={3}
                  />
                  <YAxis
                    stroke="#4a6080"
                    tick={{ fill: '#4a6080', fontSize: 11 }}
                    tickLine={false}
                    axisLine={false}
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
                  <Area
                    type="monotone"
                    dataKey="predictions"
                    stroke="#00ff9d"
                    fill="url(#predGrad)"
                    strokeWidth={2}
                  />
                  <Area
                    type="monotone"
                    dataKey="alerts"
                    stroke="#ff3b5c"
                    fill="url(#alertGrad)"
                    strokeWidth={2}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>

      {/* Patient Quick List */}
      <div className="bg-surface border border-border rounded-lg">
        <div className="px-4 py-3 border-b border-border">
          <h2 className="font-heading text-sm font-semibold flex items-center gap-2">
            <Users className="w-4 h-4 text-accent" />
            Monitored Patients
          </h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-text-muted text-xs uppercase border-b border-border">
                <th className="text-left px-4 py-3 font-medium">Patient</th>
                <th className="text-left px-4 py-3 font-medium">Temp</th>
                <th className="text-left px-4 py-3 font-medium">HR</th>
                <th className="text-left px-4 py-3 font-medium">RR</th>
                <th className="text-left px-4 py-3 font-medium">SBP</th>
                <th className="text-left px-4 py-3 font-medium">SpO2</th>
                <th className="text-left px-4 py-3 font-medium">Lactate</th>
                <th className="text-left px-4 py-3 font-medium">Risk</th>
                <th className="text-left px-4 py-3 font-medium">Updated</th>
              </tr>
            </thead>
            <tbody>
              {DEMO_PATIENTS.map((p) => (
                <tr key={p.id} className="border-b border-border/50 hover:bg-elevated/50 transition-colors cursor-pointer" onClick={() => navigate(`/patients/${p.id}`)}>
                  <td className="px-4 py-3 font-medium">{p.id}</td>
                  <td className="px-4 py-3 text-text-secondary">{p.temp}°</td>
                  <td className="px-4 py-3 text-text-secondary">{p.hr}</td>
                  <td className="px-4 py-3 text-text-secondary">{p.rr}</td>
                  <td className="px-4 py-3 text-text-secondary">{p.sbp}</td>
                  <td className="px-4 py-3 text-text-secondary">{p.spo2}%</td>
                  <td className="px-4 py-3 text-text-secondary">{p.lac}</td>
                  <td className="px-4 py-3"><RiskBadge level={p.risk} pulse={p.risk === 'critical'} /></td>
                  <td className="px-4 py-3 text-text-muted text-xs">{p.time}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
