import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { RiskBadge } from '../components/RiskBadge'
import { Search, Users } from 'lucide-react'
import { api, isDemo } from '../lib/api'
import type { RiskLevel } from '../types'

interface PatientRow {
  id: string
  bed: string
  age?: number
  risk: RiskLevel
  prob: number
  temp: number
  hr: number
  rr: number
  sbp: number
  spo2: number
  lac: number
  updated: string
}

const DEMO_PATIENTS: PatientRow[] = [
  { id: 'P-1042', bed: 'ICU-3A', age: 67, risk: 'critical', prob: 0.89, temp: 38.9, hr: 118, rr: 26, sbp: 88, spo2: 91, lac: 4.2, updated: '2m ago' },
  { id: 'P-0891', bed: 'ICU-2B', age: 54, risk: 'high', prob: 0.72, temp: 38.4, hr: 105, rr: 22, sbp: 95, spo2: 94, lac: 2.8, updated: '5m ago' },
  { id: 'P-0756', bed: 'Ward-7', age: 43, risk: 'moderate', prob: 0.38, temp: 37.8, hr: 92, rr: 20, sbp: 110, spo2: 96, lac: 1.5, updated: '8m ago' },
  { id: 'P-0623', bed: 'Ward-4', age: 71, risk: 'low', prob: 0.12, temp: 37.0, hr: 78, rr: 16, sbp: 125, spo2: 98, lac: 0.9, updated: '12m ago' },
  { id: 'P-0512', bed: 'Ward-2', age: 38, risk: 'low', prob: 0.08, temp: 36.8, hr: 72, rr: 14, sbp: 130, spo2: 99, lac: 0.7, updated: '15m ago' },
  { id: 'P-0489', bed: 'ICU-1A', age: 62, risk: 'high', prob: 0.65, temp: 39.1, hr: 112, rr: 24, sbp: 92, spo2: 92, lac: 3.1, updated: '3m ago' },
  { id: 'P-0401', bed: 'Ward-9', age: 55, risk: 'moderate', prob: 0.41, temp: 37.5, hr: 88, rr: 19, sbp: 108, spo2: 95, lac: 1.8, updated: '20m ago' },
  { id: 'P-0367', bed: 'Ward-6', age: 29, risk: 'low', prob: 0.05, temp: 36.9, hr: 68, rr: 15, sbp: 118, spo2: 99, lac: 0.6, updated: '25m ago' },
]

export function Patients() {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const [search, setSearch] = useState('')
  const [filterRisk, setFilterRisk] = useState<RiskLevel | 'all'>('all')
  const [patients, setPatients] = useState<PatientRow[]>(DEMO_PATIENTS)
  const [loading, setLoading] = useState(!isDemo)

  useEffect(() => {
    if (isDemo) return
    api.getPatients()
      .then((data) => {
        const rows: PatientRow[] = data.map((p) => ({
          id: p.id,
          bed: p.bed ?? '',
          risk: (p.riskLevel as RiskLevel) ?? 'low',
          prob: p.riskProbability ?? 0,
          temp: p.vitals?.temperature ?? 0,
          hr: p.vitals?.heart_rate ?? 0,
          rr: p.vitals?.resp_rate ?? 0,
          sbp: p.vitals?.sbp ?? 0,
          spo2: p.vitals?.spo2 ?? 0,
          lac: p.vitals?.lactate ?? 0,
          updated: p.lastUpdated ?? 'N/A',
        }))
        if (rows.length > 0) setPatients(rows)
      })
      .catch((err: unknown) => console.error('Failed to load patients:', err))
      .finally(() => setLoading(false))
  }, [])

  const filtered = patients.filter((p) => {
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
            {t('patients.title')}
          </h1>
          <p className="text-sm text-text-secondary mt-1">
            {t('patients.monitored', { n: patients.length })}
            {isDemo && <span className="ml-2 text-xs text-warning">{t('common.demoMode')}</span>}
          </p>
        </div>
      </div>

      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-3">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted" />
          <input
            type="text"
            placeholder={t('patients.searchPlaceholder')}
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full bg-surface border border-border rounded-lg pl-10 pr-4 py-2.5 text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent/50"
            aria-label={t('patients.searchLabel')}
          />
        </div>
        <div className="flex gap-2">
          {(['all', 'critical', 'high', 'moderate', 'low'] as const).map((level) => (
            <button
              key={level}
              onClick={() => setFilterRisk(level)}
              aria-pressed={filterRisk === level}
              className={`px-3 py-2 text-xs rounded-lg border transition-colors ${
                filterRisk === level
                  ? 'border-accent/50 text-accent bg-accent/10'
                  : 'border-border text-text-muted hover:text-text-secondary'
              }`}
            >
              {level === 'all' ? t('common.all') : t(`risk.${level}`)}
            </button>
          ))}
        </div>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-12">
          <div className="w-6 h-6 border-2 border-accent border-t-transparent rounded-full animate-spin" />
          <span className="ml-3 text-sm text-text-muted">{t('patients.loadingPatients')}</span>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {filtered.map((p) => (
            <button
              key={p.id}
              onClick={() => navigate(`/patients/${p.id}`)}
              aria-label={`View details for patient ${p.id}, ${p.bed}${p.age ? `, age ${p.age}` : ''}, ${p.risk} risk`}
              className="bg-surface border border-border rounded-lg p-4 hover:border-accent/30 transition-colors text-left w-full cursor-pointer"
            >
              <div className="flex items-center justify-between mb-3">
                <div>
                  <span className="font-heading font-semibold text-text-primary">{p.id}</span>
                  <span className="text-xs text-text-muted ml-2">{p.bed}</span>
                  {p.age && <span className="text-xs text-text-muted ml-2">{t('patients.age', { n: p.age })}</span>}
                </div>
                <RiskBadge level={p.risk} pulse={p.risk === 'critical'} />
              </div>
              <div className="grid grid-cols-4 gap-3 text-xs">
                <div><span className="text-text-muted block">{t('vitals.temperature')}</span><span className="text-text-secondary">{p.temp}°C</span></div>
                <div><span className="text-text-muted block">{t('vitals.heartRate')}</span><span className="text-text-secondary">{p.hr} bpm</span></div>
                <div><span className="text-text-muted block">{t('vitals.sbp')}</span><span className="text-text-secondary">{p.sbp} mmHg</span></div>
                <div><span className="text-text-muted block">{t('vitals.spo2')}</span><span className="text-text-secondary">{p.spo2}%</span></div>
                <div><span className="text-text-muted block">{t('vitals.respRate')}</span><span className="text-text-secondary">{p.rr}/min</span></div>
                <div><span className="text-text-muted block">{t('vitals.lactate')}</span><span className="text-text-secondary">{p.lac} mmol/L</span></div>
                <div><span className="text-text-muted block">{t('risk.label', { level: '' }).trim()}</span><span className="text-text-secondary">{(p.prob * 100).toFixed(0)}%</span></div>
                <div><span className="text-text-muted block">{t('dashboard.updated')}</span><span className="text-text-secondary">{p.updated}</span></div>
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
