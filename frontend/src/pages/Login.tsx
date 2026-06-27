import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Shield, LogIn, UserPlus } from 'lucide-react'
import { api, isDemo } from '../lib/api'
import { useStore } from '../stores/useStore'
import { useTranslation } from 'react-i18next'

type Mode = 'login' | 'register'

export function Login() {
  const setAuth = useStore((s) => s.setAuth)
  const navigate = useNavigate()
  const { t } = useTranslation()

  const [mode, setMode] = useState<Mode>('login')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const validate = (): string | null => {
    if (!email.trim()) return t('login.emailRequired')
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) return t('login.emailInvalid')
    if (!password) return t('login.passwordRequired')
    if (password.length < 6) return t('login.passwordTooShort')
    return null
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    const validationError = validate()
    if (validationError) {
      setError(validationError)
      return
    }
    setLoading(true)
    setError('')
    try {
      const fn = mode === 'login' ? api.login : api.register
      const res = await fn(email, password)
      setAuth(res.access_token, res.user)
      navigate('/')
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : t('login.authFailed'))
    } finally {
      setLoading(false)
    }
  }

  const handleDemoLogin = () => {
    setAuth('demo-token', { email: 'demo@sepsis-vitals.io', role: 'demo' })
    navigate('/')
  }

  return (
    <div className="min-h-screen bg-background font-mono flex items-center justify-center p-4">
      <div className="w-full max-w-sm animate-fade-in">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-accent/10 border border-accent/20 mb-4">
            <Shield className="w-7 h-7 text-accent" />
          </div>
          <h1 className="font-heading text-2xl font-bold text-text-primary">
            {t('app.title')}
          </h1>
          <p className="text-sm text-text-secondary mt-1">
            {mode === 'login' ? t('login.signIn') : t('login.createAccount')}
          </p>
        </div>

        {/* Demo Mode Button */}
        {isDemo && (
          <button
            onClick={handleDemoLogin}
            className="w-full mb-4 py-3 rounded-lg font-heading font-semibold text-sm bg-accent/10 text-accent border border-accent/30 hover:bg-accent/20 transition-colors"
          >
            {t('login.demoButton')}
          </button>
        )}

        {/* Form */}
        <form onSubmit={handleSubmit} className="bg-surface border border-border rounded-lg p-6 space-y-4">
          <div>
            <label htmlFor="email" className="block text-xs text-text-muted mb-1">
              {t('login.email')}
            </label>
            <input
              id="email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              autoComplete="email"
              placeholder={t('login.emailPlaceholder')}
              className="w-full bg-elevated border border-border rounded px-3 py-2.5 text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent/50"
            />
          </div>

          <div>
            <label htmlFor="password" className="block text-xs text-text-muted mb-1">
              {t('login.password')}
            </label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
              placeholder={t('login.passwordPlaceholder')}
              className="w-full bg-elevated border border-border rounded px-3 py-2.5 text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent/50"
            />
          </div>

          {error && (
            <p className="text-sm text-danger" role="alert">{error}</p>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-accent/10 text-accent border border-accent/30 rounded-lg py-2.5 text-sm font-medium hover:bg-accent/20 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {loading ? (
              t('common.processing')
            ) : mode === 'login' ? (
              <><LogIn className="w-4 h-4" /> {t('login.signInButton')}</>
            ) : (
              <><UserPlus className="w-4 h-4" /> {t('login.createButton')}</>
            )}
          </button>
        </form>

        {/* Mode Toggle */}
        <p className="text-center text-sm text-text-muted mt-4">
          {mode === 'login' ? (
            <>
              {t('login.noAccount')}{' '}
              <button
                onClick={() => { setMode('register'); setError('') }}
                className="text-accent hover:underline"
              >
                {t('login.createOne')}
              </button>
            </>
          ) : (
            <>
              {t('login.hasAccount')}{' '}
              <button
                onClick={() => { setMode('login'); setError('') }}
                className="text-accent hover:underline"
              >
                {t('login.signInLink')}
              </button>
            </>
          )}
        </p>
      </div>
    </div>
  )
}
