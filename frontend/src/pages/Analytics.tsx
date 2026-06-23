import { useEffect, useState } from 'react'
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

export function Analytics() {
  const weeklyData = DEMO_WEEKLY
  const riskDist = DEMO_RISK_DIST
  const [stats, setStats] = useState({
    totalPredictions: '1,110',
    alertsGenerated: '66',
    alertRate: '6.0%',
    truePositives: '48',
    ppv: '72.7%',
    avgResponse: '4.2m',
  })

  useEffect(() => {
    if (isDemo) return
    api.dashboardStats()
      .then((data) => {
        const total = data.predictions_today * 7
        const alertCount = data.active_alerts * 7
        const rate = total > 0 ? ((alertCount / total) * 100).toFixed(1) : '0.0'
        setStats((s) => ({
          ...s,
          totalPredictions: total.toLocaleString(),
          alertsGenerated: String(alertCount),
          alertRate: `${rate}%`,
          avgResponse: data.avg_response_min ? `${data.avg_response_min.toFixed(1)}m` : s.avgResponse,
        }))
      })
      .catch(() => {})
  }, [])
  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h1 className="font-heading text-2xl font-bold flex items-center gap-2">
          <BarChart3 className="w-6 h-6 text-info" />
          Analytics
        </h1>
        <p className="text-sm text-text-secondary mt-1">
          7-day performance overview
          {isDemo && <span className="ml-2 text-xs text-warning">(Demo Mode)</span>}
        </p>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="Total Predictions" value={stats.totalPredictions} sublabel="This week" color="info" />
        <StatCard label="Alerts Generated" value={stats.alertsGenerated} sublabel={`${stats.alertRate} alert rate`} color="warning" />
        <StatCard label="True Positives" value={stats.truePositives} sublabel={`${stats.ppv} PPV`} color="accent" />
        <StatCard label="Avg Response" value={stats.avgResponse} sublabel="Alert to action" color="default" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Predictions & Alerts */}
        <div className="bg-surface border border-border rounded-lg">
          <div className="px-4 py-3 border-b border-border">
            <h2 className="font-heading text-sm font-semibold">Predictions vs Alerts</h2>
          </div>
          <div className="p-4 h-[280px]">
            <ResponsiveContainer width="100%" height="100%">
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

        {/* Risk Distribution */}
        <div className="bg-surface border border-border rounded-lg">
          <div className="px-4 py-3 border-b border-border">
            <h2 className="font-heading text-sm font-semibold">Risk Distribution</h2>
          </div>
          <div className="p-4 h-[280px] flex items-center">
            <div className="w-1/2 h-full">
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
          <h2 className="font-heading text-sm font-semibold">Alert Fatigue Monitor</h2>
        </div>
        <div className="p-4 h-[240px]">
          <ResponsiveContainer width="100%" height="100%">
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
  )
}
