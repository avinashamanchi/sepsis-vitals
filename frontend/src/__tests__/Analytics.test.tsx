import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it, vi } from 'vitest'

const m = vi.hoisted(() => ({
  weeklyTrends: vi.fn(),
  riskDistribution: vi.fn(),
  dashboardStats: vi.fn(),
  modelInfo: vi.fn(),
}))
vi.mock('../lib/api', () => ({ api: m, isDemo: false }))

import { Analytics } from '../pages/Analytics'
import { Dashboard } from '../pages/Dashboard'

const pending = () => new Promise(() => {})

describe('Analytics (live mode)', () => {
  it('reports observed 7-day totals rather than extrapolations', async () => {
    m.weeklyTrends.mockResolvedValue([
      { date: '2026-10-01', predictions: 10, alerts: 1 },
      { date: '2026-10-02', predictions: 30, alerts: 3 },
    ])
    m.riskDistribution.mockResolvedValue([{ risk_level: 'high', count: 2, percentage: 50 }])
    render(<Analytics />)
    expect(await screen.findByText('40')).toBeInTheDocument()
    expect(screen.getByText('10.0% flag rate')).toBeInTheDocument()
    expect(screen.queryByText('1,110')).toBeNull()
  })

  it('shows dashes, not synthetic figures, while live data is unavailable', () => {
    m.weeklyTrends.mockReturnValue(pending())
    m.riskDistribution.mockReturnValue(pending())
    render(<Analytics />)
    expect(screen.queryByText('1,110')).toBeNull()
    expect(screen.queryByText('6.0% flag rate')).toBeNull()
    expect(screen.getAllByText('—').length).toBeGreaterThan(0)
  })
})

describe('Dashboard (live mode)', () => {
  it('never shows placeholder counts or a default AUROC before data loads', () => {
    m.modelInfo.mockReturnValue(pending())
    m.dashboardStats.mockReturnValue(pending())
    render(<MemoryRouter><Dashboard /></MemoryRouter>)
    expect(screen.queryByText('147')).toBeNull()
    expect(screen.queryByText('0.92')).toBeNull()
    expect(screen.queryByText('12')).toBeNull()
  })

  it('shows the backend 24-hour prediction count', async () => {
    m.modelInfo.mockReturnValue(pending())
    m.dashboardStats.mockResolvedValue({ patient_count: 3, active_alerts: 1, recent_predictions: 57 })
    render(<MemoryRouter><Dashboard /></MemoryRouter>)
    expect(await screen.findByText('57')).toBeInTheDocument()
  })
})
