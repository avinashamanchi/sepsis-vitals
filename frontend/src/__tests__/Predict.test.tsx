import { fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const m = vi.hoisted(() => ({ predict: vi.fn(), monitorRegister: vi.fn() }))
vi.mock('../lib/api', () => ({ api: m, isDemo: false }))

import { Predict } from '../pages/Predict'

function submit(container: HTMLElement) {
  const set = (id: string, value: string) =>
    fireEvent.change(container.querySelector(`#${id}`) as HTMLInputElement, { target: { value } })
  set('vitals-patient-id', 'research-1')
  set('vitals-heart-rate', '118')
  set('vitals-resp-rate', '24')
  set('vitals-sbp', '96')
  fireEvent.click(screen.getByRole('button', { name: 'Run Prediction' }))
}

const prediction = {
  patient_id: 'research-1', timestamp: '2026-10-09T10:00:00Z',
  risk_probability: 0.007, risk_level: 'critical', alert: true,
  rule_risk_level: 'critical', model_risk_level: 'low',
  confidence_interval: null,
  uncertainty: { method: 'none', note: 'No validated uncertainty estimate is available for this model.' },
  clinical_scores: { qsofa: 2, sirs_count: 3, news2_style: 9, shock_index: 1.23 },
  top_risk_factors: [
    { feature: 'heart_rate', label: 'Heart rate', contribution: -0.02, direction: 'lowered', current_value: 118, reference_value: 81 },
  ],
  explanation: { method: 'median substitution (per patient)', note: 'Change when replaced by the training median.' },
  data_quality: {
    warnings: [{ code: 'age_not_supplied', message: 'Age not supplied; the pipeline used 55 years.', fields: ['age_years'] }],
  },
  threshold_policy: {
    id: 'placeholder-bands-v1', status: 'engineering-placeholder: not clinically derived or approved',
    model_bands: { moderate: 0.25, high: 0.5, critical: 0.75 }, alert_cutoff: 0.6,
    alert_rule: 'alert if ...', sha256: 'f'.repeat(64),
  },
  versions: { model_version: '2.0.0', threshold_policy_id: 'placeholder-bands-v1' },
  decision_support_notice: 'Decision support for research evaluation only. This is not a diagnosis.',
  request_id: 'req-1', audit_record_id: 'rec-1',
  recommendation: 'Research output only.',
  model: {}, validation_status: 'synthetic-development', clinical_use: 'not-permitted',
}

describe('Predict', () => {
  beforeEach(() => {
    m.predict.mockReset()
  })

  it('keeps the rule-based level and the model level apart from the combined badge', async () => {
    m.predict.mockResolvedValue(prediction)
    const { container } = render(<MemoryRouter><Predict /></MemoryRouter>)
    submit(container)
    const sources = await screen.findByLabelText('Risk level sources')
    expect(sources).toHaveTextContent(/Rule-based scores\s*critical/)
    expect(sources).toHaveTextContent(/Development model\s*low/)
    expect(screen.getByText(/higher of the two levels/)).toBeInTheDocument()
    expect(screen.getByText(/not a calibrated probability/)).toBeInTheDocument()
  })

  it('always states that clinical use is not permitted', async () => {
    m.predict.mockResolvedValue({ ...prediction, clinical_use: undefined })
    const { container } = render(<MemoryRouter><Predict /></MemoryRouter>)
    submit(container)
    expect(await screen.findByTestId('clinical-use')).toHaveTextContent('Clinical use: not permitted')
    // nothing on the page can switch it on
    expect(screen.queryByRole('button', { name: /clinical/i })).toBeNull()
    expect(screen.queryByRole('checkbox', { name: /clinical/i })).toBeNull()
  })

  it('explains an unavailable model instead of showing a result', async () => {
    m.predict.mockRejectedValue(new Error('Predictions are unavailable: no usable model is installed. (model state: absent)'))
    const { container } = render(<MemoryRouter><Predict /></MemoryRouter>)
    submit(container)
    expect(await screen.findByText(/model state: absent/)).toBeInTheDocument()
    expect(screen.queryByText('Prediction Result')).toBeNull()
    expect(screen.getByTestId('no-risk')).toHaveTextContent('not a low-risk result')
  })

  it('never leaves an earlier result on screen when a new request fails', async () => {
    m.predict.mockResolvedValueOnce(prediction)
    const { container } = render(<MemoryRouter><Predict /></MemoryRouter>)
    submit(container)
    expect(await screen.findByText('Prediction Result')).toBeInTheDocument()
    m.predict.mockRejectedValueOnce(new Error('No risk calculated: the model could not score this request.'))
    submit(container)
    expect(await screen.findByText(/could not score this request/)).toBeInTheDocument()
    expect(screen.queryByText('Prediction Result')).toBeNull()
    expect(screen.queryByText('0.7%')).toBeNull()
  })

  it('shows decision-support language, honest uncertainty and data-quality warnings', async () => {
    m.predict.mockResolvedValue(prediction)
    const { container } = render(<MemoryRouter><Predict /></MemoryRouter>)
    submit(container)
    expect(await screen.findByTestId('decision-support')).toHaveTextContent('not a diagnosis')
    expect(screen.getByTestId('uncertainty')).toHaveTextContent('No validated uncertainty estimate')
    expect(screen.queryByText(/95% CI/)).toBeNull()
    expect(screen.getByTestId('data-quality')).toHaveTextContent('Age not supplied')
    expect(screen.getByTestId('attribution')).toHaveTextContent('not causal')
    expect(screen.getByTestId('traceability')).toHaveTextContent('placeholder-bands-v1')
    expect(screen.getByTestId('traceability')).toHaveTextContent('rec-1')
  })

  it('sends comorbidities with the field names the API accepts', async () => {
    m.predict.mockResolvedValue(prediction)
    const { container } = render(<MemoryRouter><Predict /></MemoryRouter>)
    fireEvent.click(screen.getByRole('checkbox', { name: 'Diabetes' }))
    submit(container)
    await screen.findByText('Prediction Result')
    expect(m.predict.mock.calls[0][0].comorbidities).toEqual({ has_diabetes: 1 })
  })
})
