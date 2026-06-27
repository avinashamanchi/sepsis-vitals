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
import { useTranslation } from 'react-i18next'

const COMORBIDITIES = [
  { key: 'hypertension', labelKey: 'predict.hypertension' },
  { key: 'diabetes', labelKey: 'predict.diabetes' },
  { key: 'ckd', labelKey: 'predict.ckd' },
  { key: 'copd', labelKey: 'predict.copd' },
  { key: 'heart_failure', labelKey: 'predict.heartFailure' },
]

/** Compute clinical scores from vitals */
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
  const { t } = useTranslation()
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
    ? probabilityToRisk(result.risk_probability * 0.85)
    : null
  const onDemandRisk: RiskLevel | null = result
    ? probabilityToRisk(result.risk_probability)
    : null

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h1 className="font-heading text-2xl font-bold flex items-center gap-2">
          <Brain className="w-6 h-6 text-accent" />
          {t('predict.title')}
        </h1>
        <p className="text-sm text-text-secondary mt-1">
          {t('predict.subtitle')}
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left column: Input */}
        <div className="space-y-4">
          <div className="bg-surface border border-border rounded-lg p-5">
            <h2 className="font-heading text-sm font-semibold mb-4">{t('predict.patientVitals')}</h2>
            <VitalsForm onSubmit={handleSubmit} loading={loading} submitLabel={t('predict.runPrediction')} />
            {error && <p className="mt-3 text-sm text-danger">{error}</p>}
          </div>

          {/* Demographics */}
          <div className="bg-surface border border-border rounded-lg p-5">
            <h2 className="font-heading text-sm font-semibold mb-3">{t('predict.demographics')}</h2>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs text-text-muted mb-1">{t('predict.age')}</label>
                <input
                  type="number"
                  min={0}
                  max={120}
                  value={age}
                  onChange={(e) => setAge(e.target.value)}
                  placeholder={t('predict.years')}
                  className="w-full bg-elevated border border-border rounded px-3 py-2 text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent/50"
                />
              </div>
              <div>
                <label className="block text-xs text-text-muted mb-1">{t('predict.sex')}</label>
                <select
                  value={sex}
                  onChange={(e) => setSex(e.target.value)}
                  className="w-full bg-elevated border border-border rounded px-3 py-2 text-sm text-text-primary focus:outline-none focus:border-accent/50"
                >
                  <option value="">--</option>
                  <option value="M">{t('predict.male')}</option>
                  <option value="F">{t('predict.female')}</option>
                </select>
              </div>
            </div>
          </div>

          {/* Comorbidities */}
          <div className="bg-surface border border-border rounded-lg p-5">
            <h2 className="font-heading text-sm font-semibold mb-3">{t('predict.comorbidities')}</h2>
            <div className="grid grid-cols-2 gap-2">
              {COMORBIDITIES.map(({ key, labelKey }) => (
                <label key={key} className="flex items-center gap-2 text-sm text-text-secondary cursor-pointer hover:text-text-primary">
                  <input
                    type="checkbox"
                    checked={comorbidities[key] ?? false}
                    onChange={(e) => setComorbidities({ ...comorbidities, [key]: e.target.checked })}
                    className="accent-accent"
                  />
                  {t(labelKey)}
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
            {t('predict.addToMonitoring')}
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
                  <h2 className="font-heading text-sm font-semibold">{t('predict.predictionResult')}</h2>
                  <RiskBadge level={result.risk_level} size="md" pulse={result.alert} />
                </div>
                <div className="text-center py-4">
                  <p className="text-5xl font-bold font-heading text-text-primary">
                    {(result.risk_probability * 100).toFixed(1)}%
                  </p>
                  <p className="text-sm text-text-secondary mt-2">{t('predict.riskProbability')}</p>
                  <p className="text-xs text-text-muted mt-1">
                    {t('predict.confidenceInterval', {
                      lower: (result.confidence_interval.lower * 100).toFixed(1),
                      upper: (result.confidence_interval.upper * 100).toFixed(1),
                    })}
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
                    <p className="text-[10px] text-text-muted uppercase tracking-wider mb-1">{t('predict.continuousMonitoring')}</p>
                    <p className="text-xs text-text-muted mb-2">{t('predict.specificityThreshold99')}</p>
                    <RiskBadge level={continuousRisk} size="sm" />
                  </div>
                  <div className="bg-surface border border-border rounded-lg p-4">
                    <p className="text-[10px] text-text-muted uppercase tracking-wider mb-1">{t('predict.clinicalAssessment')}</p>
                    <p className="text-xs text-text-muted mb-2">{t('predict.specificityThreshold95')}</p>
                    <RiskBadge level={onDemandRisk} size="sm" />
                  </div>
                </div>
              )}

              {/* Clinical Scores */}
              {scores && (
                <div className="grid grid-cols-4 gap-2">
                  {[
                    { label: t('scores.qsofa'), key: 'qSOFA', value: scores.qsofa, suffix: '/3' },
                    { label: t('scores.sirs'), key: 'SIRS', value: scores.sirs, suffix: '/4' },
                    { label: t('scores.news2'), key: 'NEWS2', value: scores.news2, suffix: '' },
                    { label: t('scores.si'), key: 'SI', value: scores.shockIndex, suffix: '' },
                  ].map((s) => (
                    <div key={s.key} className="bg-surface border border-border rounded-lg p-3 text-center">
                      <p className="text-[10px] text-text-muted">{s.label}</p>
                      <p className={clsx('text-xl font-bold font-heading', scoreColor(s.key, typeof s.value === 'number' ? s.value : 0))}>
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
                    {t('predict.riskFactors')}
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
              <p className="text-sm text-text-muted">{t('predict.emptyState')}</p>
              <p className="text-xs text-text-muted mt-1">{t('predict.emptyStateHint')}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
