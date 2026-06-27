import { useState } from 'react'
import type { Vitals } from '../types'

const VITAL_FIELDS: Array<{ key: keyof Vitals; label: string; unit: string; min: number; max: number; step: number }> = [
  { key: 'temperature', label: 'Temp', unit: '°C', min: 30, max: 45, step: 0.1 },
  { key: 'heart_rate', label: 'HR', unit: 'bpm', min: 20, max: 300, step: 1 },
  { key: 'resp_rate', label: 'RR', unit: '/min', min: 4, max: 60, step: 1 },
  { key: 'sbp', label: 'SBP', unit: 'mmHg', min: 40, max: 300, step: 1 },
  { key: 'spo2', label: 'SpO2', unit: '%', min: 50, max: 100, step: 1 },
  { key: 'gcs', label: 'GCS', unit: '/15', min: 3, max: 15, step: 1 },
  { key: 'lactate', label: 'Lactate', unit: 'mmol/L', min: 0, max: 30, step: 0.1 },
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
  const [errors, setErrors] = useState<Record<string, string>>({})
  const [formError, setFormError] = useState('')

  const validate = (): boolean => {
    const newErrors: Record<string, string> = {}
    let filledCount = 0

    if (!patientId.trim()) {
      newErrors['patientId'] = 'Patient ID is required'
    }

    for (const field of VITAL_FIELDS) {
      const raw = values[field.key]
      if (raw !== undefined && raw !== '') {
        filledCount++
        const num = parseFloat(raw)
        if (isNaN(num)) {
          newErrors[field.key] = 'Invalid number'
        } else if (num < field.min || num > field.max) {
          newErrors[field.key] = `Must be ${field.min}–${field.max}`
        }
      }
    }

    if (filledCount < 3) {
      setFormError('At least 3 vital signs are required')
      setErrors(newErrors)
      return false
    }

    setFormError('')
    setErrors(newErrors)
    return Object.keys(newErrors).length === 0
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!validate()) return
    const vitals: Record<string, number> = {}
    for (const [k, v] of Object.entries(values)) {
      if (v !== '') vitals[k] = parseFloat(v)
    }
    onSubmit(vitals, patientId.trim())
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label htmlFor="vitals-patient-id" className="block text-xs text-text-muted mb-1">
          Patient ID <span className="text-danger">*</span>
        </label>
        <input
          id="vitals-patient-id"
          type="text"
          required
          value={patientId}
          onChange={(e) => setPatientId(e.target.value)}
          placeholder="P-0001"
          className={`w-full bg-elevated border rounded px-3 py-2 text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent/50 ${errors['patientId'] ? 'border-danger' : 'border-border'}`}
        />
        {errors['patientId'] && <p className="text-xs text-danger mt-1">{errors['patientId']}</p>}
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {VITAL_FIELDS.map(({ key, label, unit, min, max, step }) => (
          <div key={key}>
            <label htmlFor={`vitals-${key.replace(/_/g, '-')}`} className="block text-xs text-text-muted mb-1">
              {label} <span className="text-text-muted/50">{unit}</span>
            </label>
            <input
              id={`vitals-${key.replace(/_/g, '-')}`}
              type="number"
              min={min}
              max={max}
              step={step}
              value={values[key] ?? ''}
              onChange={(e) => {
                setValues({ ...values, [key]: e.target.value })
                if (errors[key]) setErrors({ ...errors, [key]: '' })
              }}
              className={`w-full bg-elevated border rounded px-3 py-2 text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent/50 ${errors[key] ? 'border-danger' : 'border-border'}`}
            />
            {errors[key] && <p className="text-xs text-danger mt-1">{errors[key]}</p>}
          </div>
        ))}
      </div>

      {formError && (
        <p className="text-sm text-danger" role="alert">{formError}</p>
      )}

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
