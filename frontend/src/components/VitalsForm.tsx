import { useState } from 'react'
import type { Vitals } from '../types'

const VITAL_FIELDS: Array<{ key: keyof Vitals; label: string; unit: string; min: number; max: number; step: number }> = [
  { key: 'temperature', label: 'Temp', unit: '°C', min: 30, max: 42, step: 0.1 },
  { key: 'heart_rate', label: 'HR', unit: 'bpm', min: 30, max: 220, step: 1 },
  { key: 'resp_rate', label: 'RR', unit: '/min', min: 4, max: 60, step: 1 },
  { key: 'sbp', label: 'SBP', unit: 'mmHg', min: 40, max: 260, step: 1 },
  { key: 'spo2', label: 'SpO2', unit: '%', min: 50, max: 100, step: 1 },
  { key: 'gcs', label: 'GCS', unit: '/15', min: 3, max: 15, step: 1 },
  { key: 'lactate', label: 'Lactate', unit: 'mmol/L', min: 0, max: 20, step: 0.1 },
  { key: 'wbc', label: 'WBC', unit: 'x10^9', min: 0, max: 50, step: 0.1 },
  { key: 'procalcitonin', label: 'PCT', unit: 'ng/mL', min: 0, max: 100, step: 0.01 },
]

interface VitalsFormProps {
  onSubmit: (vitals: Record<string, number>, patientId: string) => void
  loading?: boolean
  submitLabel?: string
}

export function VitalsForm({ onSubmit, loading, submitLabel = 'Analyze' }: VitalsFormProps) {
  const [values, setValues] = useState<Record<string, string>>({})
  const [patientId, setPatientId] = useState('')

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const vitals: Record<string, number> = {}
    for (const [k, v] of Object.entries(values)) {
      if (v !== '') vitals[k] = parseFloat(v)
    }
    onSubmit(vitals, patientId || 'patient-001')
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label className="block text-xs text-text-muted mb-1">Patient ID</label>
        <input
          type="text"
          value={patientId}
          onChange={(e) => setPatientId(e.target.value)}
          placeholder="patient-001"
          className="w-full bg-elevated border border-border rounded px-3 py-2 text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent/50"
        />
      </div>

      <div className="grid grid-cols-3 gap-3">
        {VITAL_FIELDS.map(({ key, label, unit, min, max, step }) => (
          <div key={key}>
            <label className="block text-xs text-text-muted mb-1">
              {label} <span className="text-text-muted/50">{unit}</span>
            </label>
            <input
              type="number"
              min={min}
              max={max}
              step={step}
              value={values[key] ?? ''}
              onChange={(e) => setValues({ ...values, [key]: e.target.value })}
              className="w-full bg-elevated border border-border rounded px-3 py-2 text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent/50"
            />
          </div>
        ))}
      </div>

      <button
        type="submit"
        disabled={loading}
        className="w-full bg-accent/10 text-accent border border-accent/30 rounded-lg py-2.5 text-sm font-medium hover:bg-accent/20 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {loading ? 'Processing...' : submitLabel}
      </button>
    </form>
  )
}
