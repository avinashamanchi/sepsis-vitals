import { useState } from 'react'
import { Activity, ArrowLeft, ArrowRight, FlaskConical, KeyRound } from 'lucide-react'
import { Link, useNavigate } from 'react-router-dom'
import { api, isDemo } from '../lib/api'
import { useStore } from '../stores/useStore'

/** Reset links carry the token in the URL fragment, which browsers never send to servers. */
function readResetToken(): string | null {
  try {
    return new URLSearchParams(window.location.hash.slice(1)).get('reset_token')
  } catch {
    return null
  }
}

export function Login() {
  const setAuth = useStore((state) => state.setAuth)
  const navigate = useNavigate()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [info, setInfo] = useState('')
  const [loading, setLoading] = useState(false)
  const [resetToken, setResetToken] = useState<string | null>(() => (isDemo ? null : readResetToken()))
  const [otp, setOtp] = useState('')
  const [needsCode, setNeedsCode] = useState(false)
  const [enrollment, setEnrollment] = useState<{
    token: string
    secret: string
    uri: string
    codes?: string[]
  } | null>(null)
  const [enrollCode, setEnrollCode] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')

  const handleConfirmReset = async (event: React.FormEvent) => {
    event.preventDefault()
    setError('')
    if (newPassword.length < 12) {
      setError('Use at least 12 characters.')
      return
    }
    if (newPassword !== confirmPassword) {
      setError('The passwords do not match.')
      return
    }
    if (!resetToken) return
    setLoading(true)
    try {
      await api.confirmPasswordReset(resetToken, newPassword)
      // Drop the single-use token from the address bar and history.
      window.history.replaceState(null, '', window.location.pathname + window.location.search)
      setResetToken(null)
      setNewPassword('')
      setConfirmPassword('')
      setInfo('Password updated. Sign in with your new password.')
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'This reset link is invalid or has expired.')
    } finally {
      setLoading(false)
    }
  }

  const handleDemoLogin = () => {
    setAuth('demo-token', { email: 'research-demo@sepsisvitals.com', role: 'demo' })
    navigate('/dashboard')
  }

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault()
    setError('')
    setInfo('')

    if (!email.trim() || !password) {
      setError('Enter your email and password.')
      return
    }

    setLoading(true)
    try {
      const result = await api.login(email.trim(), password, needsCode ? otp.trim() : undefined)
      if (result.kind === 'mfa_required') {
        setNeedsCode(true)
        setInfo('Enter the 6-digit code from your authenticator app, or a recovery code.')
        return
      }
      if (result.kind === 'mfa_enrollment_required') {
        const started = await api.mfaEnroll(result.enrollmentToken)
        setEnrollment({ token: result.enrollmentToken, secret: started.secret, uri: started.otpauth_uri })
        return
      }
      setAuth(result.access_token, {
        email: result.user?.email ?? email.trim(),
        role: result.user?.role ?? 'nurse',
      })
      navigate('/dashboard')
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Sign in failed.')
    } finally {
      setLoading(false)
    }
  }

  const handleConfirmEnrollment = async (event: React.FormEvent) => {
    event.preventDefault()
    if (!enrollment) return
    setError('')
    setLoading(true)
    try {
      const confirmed = await api.mfaConfirm(enrollment.token, enrollCode.trim())
      setEnrollment({ ...enrollment, codes: confirmed.recovery_codes })
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'That code was not accepted.')
    } finally {
      setLoading(false)
    }
  }

  const finishEnrollment = () => {
    setEnrollment(null)
    setEnrollCode('')
    setNeedsCode(true)
    setInfo('Two-factor authentication is on. Sign in with a code from your app.')
  }

  const handleReset = async () => {
    if (!email.trim()) {
      setError('Enter your email first.')
      return
    }
    setLoading(true)
    setError('')
    try {
      await api.requestPasswordReset(email.trim())
      setInfo('If the account exists, password-reset instructions have been sent.')
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Could not request a password reset.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="grid min-h-screen bg-void text-text-primary lg:grid-cols-[.9fr_1.1fr]">
      <section className="flex min-h-screen flex-col px-5 py-6 sm:px-10 lg:px-14">
        <div className="flex items-center justify-between">
          <Link to="/" className="flex items-center gap-2.5 font-heading text-sm font-semibold">
            <span className="grid h-8 w-8 place-items-center rounded-lg border border-accent/25 bg-accent/10">
              <Activity className="h-4 w-4 text-accent" aria-hidden="true" />
            </span>
            Sepsis Vitals
          </Link>
          <Link to="/" className="flex items-center gap-1.5 text-xs text-text-muted hover:text-text-primary">
            <ArrowLeft className="h-3.5 w-3.5" aria-hidden="true" />
            Back to site
          </Link>
        </div>

        <div className="my-auto w-full max-w-md self-center py-14">
          <div className="mb-8">
            <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-accent">
              {isDemo ? 'Interactive research demo' : 'Authorized study access'}
            </p>
            <h1 className="mt-3 font-heading text-3xl font-bold tracking-tight">
              {isDemo ? 'Explore the workflow.' : 'Sign in to your study workspace.'}
            </h1>
            <p className="mt-3 text-sm leading-6 text-text-secondary">
              {isDemo
                ? 'All patients, alerts, and results in this environment are synthetic.'
                : 'Access is limited to approved research and validation partners.'}
            </p>
          </div>

          {isDemo ? (
            <button
              onClick={handleDemoLogin}
              className="group flex w-full items-center justify-between rounded-xl border border-accent/25 bg-accent/8 p-5 text-left transition-colors hover:bg-accent/12"
            >
              <span>
                <span className="block font-heading text-sm font-semibold text-accent">Open synthetic ward</span>
                <span className="mt-1 block text-xs text-text-secondary">No account or patient data required</span>
              </span>
              <ArrowRight className="h-5 w-5 text-accent transition-transform group-hover:translate-x-1" aria-hidden="true" />
            </button>
          ) : enrollment ? (
            enrollment.codes ? (
              <div className="space-y-4" aria-label="Recovery codes">
                <p className="text-sm text-text-secondary">
                  Two-factor authentication is enabled. Store these single-use recovery codes
                  somewhere safe; they are shown only once.
                </p>
                <ul className="grid grid-cols-2 gap-2 font-mono text-sm">
                  {enrollment.codes.map((code) => <li key={code}>{code}</li>)}
                </ul>
                <button
                  type="button"
                  onClick={finishEnrollment}
                  className="w-full rounded-md bg-accent px-4 py-3 text-sm font-bold text-void"
                >
                  I have stored my recovery codes
                </button>
              </div>
            ) : (
              <form onSubmit={handleConfirmEnrollment} className="space-y-4" aria-label="Set up two-factor authentication">
                <p className="text-sm text-text-secondary">
                  Your role requires two-factor authentication. Add this key to an authenticator
                  app, then enter the 6-digit code it shows.
                </p>
                <p className="break-all rounded-md border border-border bg-surface p-3 font-mono text-sm" aria-label="Authenticator key">
                  {enrollment.secret}
                </p>
                <label htmlFor="enroll-code" className="mb-1.5 block text-xs text-text-secondary">Authenticator code</label>
                <input
                  id="enroll-code"
                  inputMode="numeric"
                  autoComplete="one-time-code"
                  value={enrollCode}
                  onChange={(event) => setEnrollCode(event.target.value)}
                  required
                  className="w-full rounded-md border border-border bg-surface px-3.5 py-3 text-sm outline-none focus:border-accent/50"
                />
                {error && <p role="alert" className="text-xs leading-5 text-danger">{error}</p>}
                <button
                  type="submit"
                  disabled={loading}
                  className="w-full rounded-md bg-accent px-4 py-3 text-sm font-bold text-void disabled:opacity-50"
                >
                  Turn on two-factor authentication
                </button>
              </form>
            )
          ) : resetToken ? (
            <form onSubmit={handleConfirmReset} className="space-y-4" aria-label="Set a new password">
              <div>
                <label htmlFor="new-password" className="mb-1.5 block text-xs text-text-secondary">
                  New password (at least 12 characters)
                </label>
                <input
                  id="new-password"
                  type="password"
                  value={newPassword}
                  onChange={(event) => setNewPassword(event.target.value)}
                  autoComplete="new-password"
                  required
                  minLength={12}
                  className="w-full rounded-md border border-border bg-surface px-3.5 py-3 text-sm outline-none transition-colors focus:border-accent/50"
                />
              </div>
              <div>
                <label htmlFor="confirm-password" className="mb-1.5 block text-xs text-text-secondary">
                  Confirm new password
                </label>
                <input
                  id="confirm-password"
                  type="password"
                  value={confirmPassword}
                  onChange={(event) => setConfirmPassword(event.target.value)}
                  autoComplete="new-password"
                  required
                  minLength={12}
                  className="w-full rounded-md border border-border bg-surface px-3.5 py-3 text-sm outline-none transition-colors focus:border-accent/50"
                />
              </div>
              {error && <p role="alert" className="text-xs leading-5 text-danger">{error}</p>}
              <button
                type="submit"
                disabled={loading}
                className="flex w-full items-center justify-center gap-2 rounded-md bg-accent px-4 py-3 text-sm font-bold text-void disabled:cursor-not-allowed disabled:opacity-50"
              >
                <KeyRound className="h-4 w-4" aria-hidden="true" />
                {loading ? 'Saving…' : 'Set new password'}
              </button>
            </form>
          ) : (
            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label htmlFor="email" className="mb-1.5 block text-xs text-text-secondary">
                  Work email
                </label>
                <input
                  id="email"
                  type="email"
                  value={email}
                  onChange={(event) => setEmail(event.target.value)}
                  autoComplete="email"
                  required
                  placeholder="you@hospital.org"
                  className="w-full rounded-md border border-border bg-surface px-3.5 py-3 text-sm outline-none transition-colors placeholder:text-text-muted focus:border-accent/50"
                />
              </div>
              <div>
                <div className="mb-1.5 flex items-center justify-between">
                  <label htmlFor="password" className="text-xs text-text-secondary">Password</label>
                  <button
                    type="button"
                    onClick={handleReset}
                    disabled={loading}
                    className="text-[11px] text-accent hover:underline disabled:opacity-50"
                  >
                    Reset password
                  </button>
                </div>
                <input
                  id="password"
                  type="password"
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  autoComplete="current-password"
                  required
                  className="w-full rounded-md border border-border bg-surface px-3.5 py-3 text-sm outline-none transition-colors focus:border-accent/50"
                />
              </div>
              {needsCode && (
                <div>
                  <label htmlFor="otp" className="mb-1.5 block text-xs text-text-secondary">
                    Verification code
                  </label>
                  <input
                    id="otp"
                    inputMode="numeric"
                    autoComplete="one-time-code"
                    value={otp}
                    onChange={(event) => setOtp(event.target.value)}
                    required
                    className="w-full rounded-md border border-border bg-surface px-3.5 py-3 text-sm outline-none transition-colors focus:border-accent/50"
                  />
                </div>
              )}
              {error && <p role="alert" className="text-xs leading-5 text-danger">{error}</p>}
              {info && <p role="status" className="text-xs leading-5 text-accent">{info}</p>}
              <button
                type="submit"
                disabled={loading}
                className="flex w-full items-center justify-center gap-2 rounded-md bg-accent px-4 py-3 text-sm font-bold text-void disabled:cursor-not-allowed disabled:opacity-50"
              >
                <KeyRound className="h-4 w-4" aria-hidden="true" />
                {loading ? 'Signing in…' : 'Sign in'}
              </button>
            </form>
          )}

          <div className="mt-6 rounded-lg border border-warning/20 bg-warning/6 p-4">
            <div className="flex gap-2.5">
              <FlaskConical className="mt-0.5 h-4 w-4 shrink-0 text-warning" aria-hidden="true" />
              <p className="text-[11px] leading-5 text-text-secondary">
                <strong className="text-warning">Not for patient care.</strong> This product has
                not completed clinical validation or regulatory review. Never enter identifiable
                patient data in the public demo.
              </p>
            </div>
          </div>

          <p className="mt-6 text-center text-xs text-text-muted">
            Need partner access?{' '}
            <a href="mailto:sales@sepsisvitals.com" className="text-accent hover:underline">
              Request a pilot fit review
            </a>
          </p>
        </div>
      </section>

      <aside className="relative hidden overflow-hidden border-l border-border bg-surface/55 lg:block">
        <div className="landing-grid absolute inset-0 opacity-50" aria-hidden="true" />
        <div className="relative flex h-full items-center justify-center p-14">
          <div className="max-w-md">
            <div className="mb-7 grid h-12 w-12 place-items-center rounded-xl border border-accent/25 bg-accent/10">
              <FlaskConical className="h-5 w-5 text-accent" aria-hidden="true" />
            </div>
            <blockquote className="font-heading text-2xl font-semibold leading-snug tracking-tight">
              “A credible medical AI company shows the limitations before it shows the dashboard.”
            </blockquote>
            <p className="mt-6 text-sm leading-6 text-text-secondary">
              Every output in the research workspace is labeled with its intended use,
              validation status, and model provenance.
            </p>
          </div>
        </div>
      </aside>
    </main>
  )
}
