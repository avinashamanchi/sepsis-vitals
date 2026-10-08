import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const m = vi.hoisted(() => ({
  login: vi.fn(),
  requestPasswordReset: vi.fn(),
  confirmPasswordReset: vi.fn(),
}))
vi.mock('../lib/api', () => ({ api: m, isDemo: false }))

import { Login } from '../pages/Login'
import { useStore } from '../stores/useStore'

const renderLogin = () => render(<MemoryRouter><Login /></MemoryRouter>)

beforeEach(() => {
  Object.values(m).forEach((fn) => fn.mockReset())
  window.history.replaceState(null, '', '/login')
  useStore.setState({ token: null, user: null })
})

describe('sign in', () => {
  it('stores the session on success', async () => {
    m.login.mockResolvedValue({ access_token: 'abc', user: { email: 'n@h.org', role: 'nurse' } })
    const { container } = renderLogin()
    fireEvent.change(screen.getByLabelText('Work email'), { target: { value: 'n@h.org' } })
    fireEvent.change(screen.getByLabelText('Password'), { target: { value: 'pw' } })
    fireEvent.submit(container.querySelector('form')!)
    await waitFor(() => expect(useStore.getState().token).toBe('abc'))
  })

  it('shows the server error and stays signed out on failure', async () => {
    m.login.mockRejectedValue(new Error('Invalid email or password'))
    const { container } = renderLogin()
    fireEvent.change(screen.getByLabelText('Work email'), { target: { value: 'n@h.org' } })
    fireEvent.change(screen.getByLabelText('Password'), { target: { value: 'bad' } })
    fireEvent.submit(container.querySelector('form')!)
    expect(await screen.findByRole('alert')).toHaveTextContent('Invalid email or password')
    expect(useStore.getState().token).toBeNull()
  })
})

describe('password reset', () => {
  it('requests a reset with a response that does not reveal the account', async () => {
    m.requestPasswordReset.mockResolvedValue({ detail: 'ok' })
    renderLogin()
    fireEvent.change(screen.getByLabelText('Work email'), { target: { value: 'n@h.org' } })
    fireEvent.click(screen.getByRole('button', { name: 'Reset password' }))
    expect(await screen.findByRole('status')).toHaveTextContent(/If the account exists/)
    expect(m.requestPasswordReset).toHaveBeenCalledWith('n@h.org')
  })

  it('consumes the token from the URL fragment and removes it', async () => {
    window.history.replaceState(null, '', '/login#reset_token=tok-123')
    m.confirmPasswordReset.mockResolvedValue({ detail: 'ok' })
    renderLogin()
    fireEvent.change(screen.getByLabelText(/New password/), { target: { value: 'a-long-new-passphrase' } })
    fireEvent.change(screen.getByLabelText('Confirm new password'), { target: { value: 'a-long-new-passphrase' } })
    fireEvent.submit(screen.getByRole('form', { name: 'Set a new password' }))
    await waitFor(() =>
      expect(m.confirmPasswordReset).toHaveBeenCalledWith('tok-123', 'a-long-new-passphrase'),
    )
    expect(window.location.hash).toBe('')
    expect(await screen.findByText(/Password updated/)).toBeInTheDocument()
  })

  it('rejects mismatched passwords without calling the API', async () => {
    window.history.replaceState(null, '', '/login#reset_token=tok-123')
    renderLogin()
    fireEvent.change(screen.getByLabelText(/New password/), { target: { value: 'a-long-new-passphrase' } })
    fireEvent.change(screen.getByLabelText('Confirm new password'), { target: { value: 'something-else-entirely' } })
    fireEvent.submit(screen.getByRole('form', { name: 'Set a new password' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('do not match')
    expect(m.confirmPasswordReset).not.toHaveBeenCalled()
  })
})
