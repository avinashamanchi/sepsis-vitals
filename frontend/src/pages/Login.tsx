import { useState } from 'react'
import { Activity, ArrowLeft, ArrowRight, FlaskConical, KeyRound } from 'lucide-react'
import { Link, useNavigate } from 'react-router-dom'
import { api, isDemo } from '../lib/api'
import { useStore } from '../stores/useStore'

export function Login() {
  const setAuth = useStore((state) => state.setAuth)
  const navigate = useNavigate()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [info, setInfo] = useState('')
  const [loading, setLoading] = useState(false)

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
      const result = await api.login(email.trim(), password)
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
