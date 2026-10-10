import { render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const m = vi.hoisted(() => ({
  getPatients: vi.fn(), weeklyTrends: vi.fn(), modelInfo: vi.fn(), dashboardStats: vi.fn(),
}))
vi.mock('../lib/api', () => ({ api: m, isDemo: false }))

import { Dashboard } from '../pages/Dashboard'

const DEMO_IDS = ['P-1042', 'P-0891', 'P-0756', 'P-0623', 'P-0512']

function renderPage() {
  return render(<MemoryRouter><Dashboard /></MemoryRouter>)
}

describe('Dashboard (live mode) shows no fabricated data (N41)', () => {
  beforeEach(() => {
    m.modelInfo.mockResolvedValue({ model_name: 'GradientBoosting', metrics: {} })
    m.dashboardStats.mockResolvedValue({ patient_count: 2, active_alerts: 0, recent_predictions: 1 })
    m.weeklyTrends.mockResolvedValue([])
  })

  it('lists only observed patients from the API, with gaps shown as gaps', async () => {
    m.getPatients.mockResolvedValue([
      { id: 'p-1', external_id: 'MRN-LIVE-1', site_id: 'A', age_years: 60, sex: 'F',
        latest_vitals: { heart_rate: 112, resp_rate: 26 }, latest_risk_level: null,
        latest_recorded_at: '2026-10-09T10:00:00Z' },
      { id: 'p-2', external_id: 'MRN-NEVER-SEEN', site_id: 'A', age_years: 40, sex: 'M',
        latest_vitals: null, latest_risk_level: null, latest_recorded_at: null },
    ])
    renderPage()
    const row = (await screen.findByText('MRN-LIVE-1')).closest('tr') as HTMLElement
    expect(within(row).getByText('112')).toBeInTheDocument()
    expect(within(row).getByText('Not scored')).toBeInTheDocument()
    expect(within(row).getAllByText('—').length).toBeGreaterThanOrEqual(4) // temp, sbp, spo2, lactate
    expect(screen.queryByText('MRN-NEVER-SEEN')).toBeNull()
    for (const id of DEMO_IDS) expect(screen.queryByText(id)).toBeNull()
  })

  it('says so when there is nothing to show, instead of inventing a trend or patients', async () => {
    m.getPatients.mockResolvedValue([])
    renderPage()
    expect(await screen.findByTestId('patients-empty')).toHaveTextContent('No observed patients yet')
    expect(await screen.findByTestId('trend-empty')).toBeInTheDocument()
    for (const id of DEMO_IDS) expect(screen.queryByText(id)).toBeNull()
  })

  it('reports a loading failure rather than falling back to demo rows', async () => {
    m.getPatients.mockRejectedValue(new Error('network down'))
    renderPage()
    expect(await screen.findByTestId('patients-empty')).toHaveTextContent('could not be loaded')
    for (const id of DEMO_IDS) expect(screen.queryByText(id)).toBeNull()
  })
})
