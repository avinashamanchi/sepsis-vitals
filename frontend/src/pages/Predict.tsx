import { useState } from 'react'
import { VitalsForm } from '../components/VitalsForm'
import { RiskBadge } from '../components/RiskBadge'
import type { Prediction } from '../types'
import { api } from '../lib/api'
import { Brain, TrendingUp } from 'lucide-react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from 'recharts'

export function Predict() {
  const [result, setResult] = useState<Prediction | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleSubmit = async (vitals: Record<string, number>, patientId: string) => {
    setLoading(true)
    setError('')
    try {
      const res = await api.predict({ vitals, patient_id: patientId }) as Prediction
      setResult(res)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  const featureColors: Record<string, string> = {
    procalcitonin: '#ff3b5c',
    lactate: '#ff6b35',
    temperature: '#ffb830',
    heart_rate: '#38b4ff',
    wbc: '#00ff9d',
  }

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h1 className="font-heading text-2xl font-bold flex items-center gap-2">
          <Brain className="w-6 h-6 text-accent" />
          AI Prediction
        </h1>
        <p className="text-sm text-text-secondary mt-1">
          ML-powered sepsis risk prediction with SHAP explanations
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-surface border border-border rounded-lg p-5">
          <h2 className="font-heading text-sm font-semibold mb-4">Patient Vitals</h2>
          <VitalsForm onSubmit={handleSubmit} loading={loading} submitLabel="Run Prediction" />
          {error && <p className="mt-3 text-sm text-danger">{error}</p>}
        </div>

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
              <p className="text-xs text-text-muted mt-1">GradientBoosting model, AUROC 0.88</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
