import { fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const m = vi.hoisted(() => ({ score: vi.fn() }))
vi.mock('../lib/api', () => ({ api: m, isDemo: false }))

import { ScoreLab } from '../pages/ScoreLab'

const result = {
  qsofa: 1, sirs_count: 2, news2_style: 6, shock_index: 1.47, uva: 5,
  uva_risk_band: 'high', uva_missing_inputs: ['temperature', 'spo2', 'gcs'],
  uva_outcome: 'in-hospital mortality, adult inpatients, sub-Saharan Africa (derivation cohort)',
  uva_reference: 'Moore CC et al. BMJ Glob Health 2017;2:e000344 (doi:10.1136/bmjgh-2017-000344)',
  risk_level: 'high', alert_flag: true, explanations: [],
}

function submit(container: HTMLElement) {
  const set = (id: string, value: string) =>
    fireEvent.change(container.querySelector(`#${id}`) as HTMLInputElement, { target: { value } })
  set('vitals-patient-id', 'synthetic-uva')
  set('vitals-heart-rate', '125')
  set('vitals-resp-rate', '31')
  set('vitals-sbp', '85')
  fireEvent.click(screen.getByRole('button', { name: 'Calculate Scores' }))
}

describe('ScoreLab', () => {
  beforeEach(() => m.score.mockReset())

  it('shows the published UVA band, what was not measured, and its source', async () => {
    m.score.mockResolvedValue(result)
    const { container } = render(<MemoryRouter><ScoreLab /></MemoryRouter>)
    submit(container)
    expect(await screen.findByText('5/13')).toBeInTheDocument()
    expect(screen.getByText(/high mortality-risk band \(not measured: temperature, spo2, gcs\)/)).toBeInTheDocument()
    expect(screen.getByTestId('uva-source')).toHaveTextContent('bmjgh-2017-000344')
    expect(screen.getByTestId('uva-source')).toHaveTextContent('not a diagnosis')
  })

  it('sends HIV status only when it is recorded', async () => {
    m.score.mockResolvedValue(result)
    const { container } = render(<MemoryRouter><ScoreLab /></MemoryRouter>)
    submit(container)
    await screen.findByText('5/13')
    expect(m.score.mock.calls[0][0]).not.toHaveProperty('hiv_positive')
    fireEvent.change(screen.getByLabelText('HIV status (used only by the UVA score)'), { target: { value: 'positive' } })
    submit(container)
    await screen.findByText('5/13')
    expect(m.score.mock.calls[1][0].hiv_positive).toBe(true)
  })
})
