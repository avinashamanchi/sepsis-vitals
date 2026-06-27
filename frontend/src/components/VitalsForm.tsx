import { useState } from 'react'
import type { Vitals } from '../types'
import { useTranslation } from 'react-i18next'

interface VitalFieldDef {
  key: keyof Vitals
  labelKey: string
  unitKey: string
  min: number
  max: number
  step: number
}

const VITAL_FIELDS: VitalFieldDef[] = [
  { key: 'temperature', labelKey: 'vitals.temperature', unitKey: 'vitals.units.celsius', min: 30, max: 45, step: 0.1 },
  { key: 'heart_rate', labelKey: 'vitals.heartRate', unitKey: 'vitals.units.bpm', min: 20, max: 300, step: 1 },
  { key: 'resp_rate', labelKey: 'vitals.respRate', unitKey: 'vitals.units.perMin', min: 4, max: 60, step: 1 },
  { key: 'sbp', labelKey: 'vitals.sbp', unitKey: 'vitals.units.mmHg', min: 40, max: 300, step: 1 },
  { key: 'spo2', labelKey: 'vitals.spo2', unitKey: 'vitals.units.percent', min: 50, max: 100, step: 1 },
  { key: 'gcs', labelKey: 'vitals.gcs', unitKey: 'vitals.units.outOf15', min: 3, max: 15, step: 1 },
  { key: 'lactate', labelKey: 'vitals.lactate', unitKey: 'vitals.units.mmolL', min: 0, max: 30, step: 0.1 },
  { key: 'wbc', labelKey: 'vitals.wbc', unitKey: 'vitals.units.x10e9', min: 0, max: 50, step: 0.1 },
  { key: 'procalcitonin', labelKey: 'vitals.pct', unitKey: 'vitals.units.ngMl', min: 0, max: 100, step: 0.01 },
]

interface VitalsFormProps {
  onSubmit: (vitals: Record<string, number>, patientId: string) => void
  loading?: boolean
  submitLabel?: string
}

export function VitalsForm({ onSubmit, loading, submitLabel }: VitalsFormProps) {
  const { t } = useTranslation()
  const resolvedSubmitLabel = submitLabel ?? t('vitalsForm.analyze')

  const [values, setValues] = useState<Record<string, string>>({})
  const [patientId, setPatientId] = useState('')
  const [errors, setErrors] = useState<Record<string, string>>({})
  const [formError, setFormError] = useState('')

  const validate = (): boolean => {
    const newErrors: Record<string, string> = {}
    let filledCount = 0

    if (!patientId.trim()) {
      newErrors['patientId'] = t('vitalsForm.patientIdRequired')
    }

    for (const field of VITAL_FIELDS) {
      const raw = values[field.key]
      if (raw !== undefined && raw !== '') {
        filledCount++
        const num = parseFloat(raw)
        if (isNaN(num)) {
          newErrors[field.key] = t('vitalsForm.invalidNumber')
        } else if (num < field.min || num > field.max) {
          newErrors[field.key] = t('vitalsForm.mustBeRange', { min: field.min, max: field.max })
        }
      }
    }

    if (filledCount < 3) {
      setFormError(t('vitalsForm.minVitals'))
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
          {t('vitalsForm.patientId')} <span className="text-danger">*</span>
        </label>
        <input
          id="vitals-patient-id"
          type="text"
          required
          value={patientId}
          onChange={(e) => setPatientId(e.target.value)}
          placeholder={t('vitalsForm.patientIdPlaceholder')}
          className={`w-full bg-elevated border rounded px-3 py-2 text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent/50 ${errors['patientId'] ? 'border-danger' : 'border-border'}`}
        />
        {errors['patientId'] && <p className="text-xs text-danger mt-1">{errors['patientId']}</p>}
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {VITAL_FIELDS.map(({ key, labelKey, unitKey, min, max, step }) => (
          <div key={key}>
            <label htmlFor={`vitals-${key.replace(/_/g, '-')}`} className="block text-xs text-text-muted mb-1">
              {t(labelKey)} <span className="text-text-muted/50">{t(unitKey)}</span>
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
        {loading ? t('common.processing') : resolvedSubmitLabel}
      </button>
    </form>
  )
}
