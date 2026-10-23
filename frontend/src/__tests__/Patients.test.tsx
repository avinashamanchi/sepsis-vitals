import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it, vi } from 'vitest'

const m = vi.hoisted(() => ({ getPatients: vi.fn() }))
vi.mock('../lib/api', () => ({ api: m, isDemo: false }))

import { Patients } from '../pages/Patients'

const renderPage = () => render(<MemoryRouter><Patients /></MemoryRouter>)

const unobserved = {
  id: 'p-1', external_id: 'MRN-1', site_id: 'A', age_years: 50, sex: 'F',
  latest_vitals: null, latest_risk_level: null, latest_recorded_at: null,
}

describe('Patients (live mode)', () => {
  it('never shows demo patients, even when the site has none', async () => {
    m.getPatients.mockResolvedValue([])
    renderPage()
    expect(await screen.findByText(/No patients match/)).toBeInTheDocument()
    expect(screen.queryByText('P-1042')).toBeNull()
  })

  it('shows an error instead of demo rows when loading fails', async () => {
    m.getPatients.mockRejectedValue(new Error('network down'))
    renderPage()
    expect(await screen.findByRole('alert')).toHaveTextContent(/could not be loaded/i)
    expect(screen.queryByText('P-1042')).toBeNull()
  })

  it('marks an unobserved patient as not scored, with no invented vitals', async () => {
    m.getPatients.mockResolvedValue([unobserved])
    renderPage()
    const card = await screen.findByRole('button', { name: /not yet scored/ })
    expect(card).toHaveTextContent('Not scored')
    expect(card).toHaveTextContent('—')
    expect(card).not.toHaveTextContent('0 bpm')
    expect(card).not.toHaveTextContent(/\blow\b/i)
  })

  it('shows the latest observed risk and vitals', async () => {
    m.getPatients.mockResolvedValue([{
      ...unobserved, latest_risk_level: 'high',
      latest_vitals: { heart_rate: 125, temperature: 39.1 },
      latest_recorded_at: '2026-10-08T10:00:00Z',
    }])
    renderPage()
    const card = await screen.findByRole('button', { name: /high risk/ })
    expect(card).toHaveTextContent('125 bpm')
    expect(card).toHaveTextContent('39.1°C')
  })
})
