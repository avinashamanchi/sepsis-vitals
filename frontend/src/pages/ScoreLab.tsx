import { useState } from 'react'
import { VitalsForm } from '../components/VitalsForm'
import { RiskBadge } from '../components/RiskBadge'
import type { RiskLevel, ScoreResult } from '../types'
import { api } from '../lib/api'
import { Calculator } from 'lucide-react'

export function ScoreLab() {
  const [result, setResult] = useState<ScoreResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleSubmit = async (vitals: Record<string, number>, _patientId: string) => {
    setLoading(true)
    setError('')
    try {
      const res = await api.score(vitals) as ScoreResult
      setResult(res)
    } catch (e: any) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h1 className="font-heading text-2xl font-bold flex items-center gap-2">
          <Calculator className="w-6 h-6 text-accent" />
          Score Lab
        </h1>
        <p className="text-sm text-text-secondary mt-1">
          Calculate qSOFA, SIRS, NEWS2, Shock Index, and UVA scores
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-surface border border-border rounded-lg p-5">
          <h2 className="font-heading text-sm font-semibold mb-4">Enter Vitals</h2>
          <VitalsForm onSubmit={handleSubmit} loading={loading} submitLabel="Calculate Scores" />
          {error && (
            <p className="mt-3 text-sm text-danger">{error}</p>
          )}
        </div>

        <div className="space-y-4">
          {result ? (
            <>
              <div className="bg-surface border border-border rounded-lg p-5">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="font-heading text-sm font-semibold">Risk Assessment</h2>
                  <RiskBadge level={result.risk_level as RiskLevel} size="md" pulse={result.alert_flag} />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  {[
                    { label: 'qSOFA', value: `${result.qsofa}/3`, desc: result.qsofa >= 2 ? 'High risk' : 'Low risk' },
                    { label: 'SIRS', value: `${result.sirs_count}/3`, desc: result.sirs_count >= 2 ? 'Criteria met' : 'Below threshold' },
                    { label: 'NEWS2', value: `${result.news2_style}`, desc: result.news2_style >= 7 ? 'High' : result.news2_style >= 5 ? 'Medium' : 'Low' },
                    { label: 'Shock Index', value: result.shock_index?.toFixed(2) ?? 'N/A', desc: (result.shock_index ?? 0) >= 1.0 ? 'Elevated' : 'Normal' },
                    { label: 'UVA', value: `${result.uva}`, desc: result.uva >= 4 ? 'High mortality' : 'Lower risk' },
                  ].map(({ label, value, desc }) => (
                    <div key={label} className="bg-elevated rounded-lg p-3">
                      <p className="text-xs text-text-muted">{label}</p>
                      <p className="text-xl font-bold font-heading mt-1">{value}</p>
                      <p className="text-xs text-text-secondary mt-1">{desc}</p>
                    </div>
                  ))}
                </div>
              </div>

              {result.explanations.length > 0 && (
                <div className="bg-surface border border-border rounded-lg p-5">
                  <h2 className="font-heading text-sm font-semibold mb-3">Clinical Notes</h2>
                  <ul className="space-y-1.5">
                    {result.explanations.map((exp, i) => (
                      <li key={i} className="text-sm text-text-secondary flex items-start gap-2">
                        <span className="text-accent mt-0.5">-</span>
                        {exp}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </>
          ) : (
            <div className="bg-surface border border-border rounded-lg p-8 text-center">
              <Calculator className="w-8 h-8 text-text-muted mx-auto mb-3" />
              <p className="text-sm text-text-muted">Enter vital signs to calculate clinical scores</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
