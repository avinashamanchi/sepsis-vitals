import { useState } from 'react'
import { RiskBadge } from '../components/RiskBadge'
import { Search, Users } from 'lucide-react'
import type { RiskLevel } from '../types'

const MOCK_PATIENTS = [
  { id: 'P-1042', bed: 'ICU-3A', age: 67, risk: 'critical' as RiskLevel, prob: 0.89, temp: 38.9, hr: 118, rr: 26, sbp: 88, spo2: 91, lac: 4.2, updated: '2m ago' },
  { id: 'P-0891', bed: 'ICU-2B', age: 54, risk: 'high' as RiskLevel, prob: 0.72, temp: 38.4, hr: 105, rr: 22, sbp: 95, spo2: 94, lac: 2.8, updated: '5m ago' },
  { id: 'P-0756', bed: 'Ward-7', age: 43, risk: 'moderate' as RiskLevel, prob: 0.38, temp: 37.8, hr: 92, rr: 20, sbp: 110, spo2: 96, lac: 1.5, updated: '8m ago' },
  { id: 'P-0623', bed: 'Ward-4', age: 71, risk: 'low' as RiskLevel, prob: 0.12, temp: 37.0, hr: 78, rr: 16, sbp: 125, spo2: 98, lac: 0.9, updated: '12m ago' },
  { id: 'P-0512', bed: 'Ward-2', age: 38, risk: 'low' as RiskLevel, prob: 0.08, temp: 36.8, hr: 72, rr: 14, sbp: 130, spo2: 99, lac: 0.7, updated: '15m ago' },
  { id: 'P-0489', bed: 'ICU-1A', age: 62, risk: 'high' as RiskLevel, prob: 0.65, temp: 39.1, hr: 112, rr: 24, sbp: 92, spo2: 92, lac: 3.1, updated: '3m ago' },
  { id: 'P-0401', bed: 'Ward-9', age: 55, risk: 'moderate' as RiskLevel, prob: 0.41, temp: 37.5, hr: 88, rr: 19, sbp: 108, spo2: 95, lac: 1.8, updated: '20m ago' },
  { id: 'P-0367', bed: 'Ward-6', age: 29, risk: 'low' as RiskLevel, prob: 0.05, temp: 36.9, hr: 68, rr: 15, sbp: 118, spo2: 99, lac: 0.6, updated: '25m ago' },
]

export function Patients() {
  const [search, setSearch] = useState('')
  const [filterRisk, setFilterRisk] = useState<RiskLevel | 'all'>('all')

  const filtered = MOCK_PATIENTS.filter((p) => {
    if (search && !p.id.toLowerCase().includes(search.toLowerCase()) && !p.bed.toLowerCase().includes(search.toLowerCase())) return false
    if (filterRisk !== 'all' && p.risk !== filterRisk) return false
    return true
  })

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="font-heading text-2xl font-bold flex items-center gap-2">
            <Users className="w-6 h-6 text-accent" />
            Patients
          </h1>
          <p className="text-sm text-text-secondary mt-1">{MOCK_PATIENTS.length} patients monitored</p>
        </div>
      </div>

      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-3">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted" />
          <input
            type="text"
            placeholder="Search patient ID or bed..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full bg-surface border border-border rounded-lg pl-10 pr-4 py-2.5 text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent/50"
          />
        </div>
        <div className="flex gap-2">
          {(['all', 'critical', 'high', 'moderate', 'low'] as const).map((level) => (
            <button
              key={level}
              onClick={() => setFilterRisk(level)}
              className={`px-3 py-2 text-xs rounded-lg border transition-colors ${
                filterRisk === level
                  ? 'border-accent/50 text-accent bg-accent/10'
                  : 'border-border text-text-muted hover:text-text-secondary'
              }`}
            >
              {level === 'all' ? 'All' : level.charAt(0).toUpperCase() + level.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Patient Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {filtered.map((p) => (
          <div
            key={p.id}
            className="bg-surface border border-border rounded-lg p-4 hover:border-border-bright transition-colors"
          >
            <div className="flex items-center justify-between mb-3">
              <div>
                <span className="font-heading font-semibold text-text-primary">{p.id}</span>
                <span className="text-xs text-text-muted ml-2">{p.bed}</span>
                <span className="text-xs text-text-muted ml-2">Age {p.age}</span>
              </div>
              <RiskBadge level={p.risk} pulse={p.risk === 'critical'} />
            </div>
            <div className="grid grid-cols-4 gap-3 text-xs">
              <div>
                <span className="text-text-muted block">Temp</span>
                <span className="text-text-secondary">{p.temp}°C</span>
              </div>
              <div>
                <span className="text-text-muted block">HR</span>
                <span className="text-text-secondary">{p.hr} bpm</span>
              </div>
              <div>
                <span className="text-text-muted block">SBP</span>
                <span className="text-text-secondary">{p.sbp} mmHg</span>
              </div>
              <div>
                <span className="text-text-muted block">SpO2</span>
                <span className="text-text-secondary">{p.spo2}%</span>
              </div>
              <div>
                <span className="text-text-muted block">RR</span>
                <span className="text-text-secondary">{p.rr}/min</span>
              </div>
              <div>
                <span className="text-text-muted block">Lactate</span>
                <span className="text-text-secondary">{p.lac} mmol/L</span>
              </div>
              <div>
                <span className="text-text-muted block">Risk</span>
                <span className="text-text-secondary">{(p.prob * 100).toFixed(0)}%</span>
              </div>
              <div>
                <span className="text-text-muted block">Updated</span>
                <span className="text-text-secondary">{p.updated}</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
