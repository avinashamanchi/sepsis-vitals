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
  confidence_interval: { lower: 0.001, upper: 0.02 },
  clinical_scores: { qsofa: 2, sirs_count: 3, news2_style: 9, shock_index: 1.23 },
  top_risk_factors: [], recommendation: 'Research output only.',
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
  })
})
