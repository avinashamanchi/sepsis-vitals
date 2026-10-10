import { render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { describe, expect, it, vi } from 'vitest'

const m = vi.hoisted(() => ({ patientTrend: vi.fn() }))
vi.mock('../lib/api', () => ({ api: m, isDemo: false }))

import { PatientDetail } from '../pages/PatientDetail'

function renderPage(id = 'p-1') {
  return render(
    <MemoryRouter initialEntries={[`/patients/${id}`]}>
      <Routes>
        <Route path="/patients/:id" element={<PatientDetail />} />
      </Routes>
    </MemoryRouter>,
  )
}

const riskBadge = () => screen.queryByLabelText(/^Risk level/)

describe('PatientDetail (N42)', () => {
  it('shows "Not scored", not a low-risk badge, when the patient has no scores', async () => {
    m.patientTrend.mockResolvedValue({ trend: [] })
    renderPage()
    expect(await screen.findByTestId('not-scored')).toHaveTextContent('Not scored')
    expect(riskBadge()).toBeNull()
  })

  it('shows the API error and no invented risk when loading fails', async () => {
    m.patientTrend.mockRejectedValue(new Error('No data for patient p-1'))
    renderPage()
    expect(await screen.findByText('No data for patient p-1')).toBeInTheDocument()
    expect(screen.getByTestId('not-scored')).toBeInTheDocument()
    expect(riskBadge()).toBeNull()
  })

  it('uses the risk level the API reports for the latest score', async () => {
    m.patientTrend.mockResolvedValue({
      trend: [
        { timestamp: '2026-10-09T09:00:00Z', risk_probability: 0.01, risk_level: 'critical', vitals: { heart_rate: 118 } },
      ],
    })
    renderPage()
    expect(await screen.findByLabelText(/^Risk level/)).toHaveTextContent('Critical')
    expect(screen.queryByTestId('not-scored')).toBeNull()
  })
})
