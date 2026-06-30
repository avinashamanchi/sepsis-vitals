import { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { Shield, LogIn, UserPlus } from 'lucide-react'
import { isDemo } from '../lib/api'
import { signInWithGoogle, signInWithEmail, signUpWithEmail, resetPassword } from '../lib/auth'
import { useStore } from '../stores/useStore'
import { useTranslation } from 'react-i18next'
import { FirebaseError } from 'firebase/app'

type Mode = 'login' | 'register'

export function Login() {
  const setAuth = useStore((s) => s.setAuth)
  const navigate = useNavigate()
  const { t } = useTranslation()

  const [mode, setMode] = useState<Mode>('login')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [error, setError] = useState('')
  const [info, setInfo] = useState('')
  const [loading, setLoading] = useState(false)

  function firebaseErrorMessage(err: unknown): string {
    if (err instanceof FirebaseError) {
      switch (err.code) {
        case 'auth/email-already-in-use': return t('login.emailInUse')
        case 'auth/weak-password': return t('login.weakPassword')
        case 'auth/user-not-found':
        case 'auth/wrong-password':
        case 'auth/invalid-credential': return t('login.authFailed')
        case 'auth/too-many-requests': return 'Too many attempts. Please try again later.'
        default: return t('login.firebaseError')
      }
    }
    return err instanceof Error ? err.message : t('login.firebaseError')
  }

  async function handleFirebaseUser(user: { email: string | null; displayName: string | null; photoURL: string | null; getIdToken: () => Promise<string> }) {
    const token = await user.getIdToken()
    setAuth(token, {
      email: user.email ?? '',
      role: 'user',
      displayName: user.displayName ?? undefined,
      photoURL: user.photoURL ?? undefined,
    })
    navigate('/dashboard')
  }

  const handleGoogle = async () => {
    setLoading(true)
    setError('')
    try {
      const user = await signInWithGoogle()
      await handleFirebaseUser(user)
    } catch (err: unknown) {
      setError(firebaseErrorMessage(err))
    } finally {
      setLoading(false)
    }
  }

  const validate = (): string | null => {
    if (!email.trim()) return t('login.emailRequired')
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) return t('login.emailInvalid')
    if (!password) return t('login.passwordRequired')
    if (password.length < 6) return t('login.passwordTooShort')
    if (mode === 'register' && password !== confirmPassword) return t('login.passwordMismatch')
    return null
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    const validationError = validate()
    if (validationError) { setError(validationError); return }

    if (isDemo) {
      setAuth('demo-token', { email, role: 'demo' })
      navigate('/dashboard')
      return
    }

    setLoading(true)
    setError('')
    try {
      const user = mode === 'login'
        ? await signInWithEmail(email, password)
        : await signUpWithEmail(email, password)
      await handleFirebaseUser(user)
    } catch (err: unknown) {
      setError(firebaseErrorMessage(err))
    } finally {
      setLoading(false)
    }
  }

  const handleForgotPassword = async () => {
    if (!email.trim()) { setError(t('login.emailRequired')); return }
    try {
      await resetPassword(email)
      setInfo(t('login.resetSent'))
      setError('')
    } catch (err: unknown) {
      setError(firebaseErrorMessage(err))
    }
  }

  const handleDemoLogin = () => {
    setAuth('demo-token', { email: 'demo@sepsis-vitals.io', role: 'demo' })
    navigate('/dashboard')
  }

  return (
    <div className="min-h-screen bg-void font-mono flex items-center justify-center p-4">
      <div className="w-full max-w-sm animate-fade-in">
        {/* Header */}
        <div className="text-center mb-8">
          <Link to="/" className="inline-block">
            <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-accent/10 border border-accent/20 mb-4">
              <Shield className="w-7 h-7 text-accent" />
            </div>
          </Link>
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

        {/* Google Sign-In */}
        {!isDemo && (
          <>
            <button
              onClick={handleGoogle}
              disabled={loading}
              className="w-full mb-4 py-3 rounded-lg font-heading font-semibold text-sm bg-surface text-text-primary border border-border hover:border-accent/50 transition-colors flex items-center justify-center gap-3 disabled:opacity-50"
            >
              <svg viewBox="0 0 24 24" className="w-5 h-5" aria-hidden="true">
                <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z"/>
                <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
              </svg>
              {t('login.googleSignIn')}
            </button>

            <div className="flex items-center gap-3 mb-4">
              <div className="flex-1 h-px bg-border" />
              <span className="text-xs text-text-muted">{t('login.orEmail')}</span>
              <div className="flex-1 h-px bg-border" />
            </div>
          </>
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

          {mode === 'register' && (
            <div>
              <label htmlFor="confirmPassword" className="block text-xs text-text-muted mb-1">
                {t('login.confirmPassword')}
              </label>
              <input
                id="confirmPassword"
                type="password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                required
                autoComplete="new-password"
                placeholder={t('login.confirmPlaceholder')}
                className="w-full bg-elevated border border-border rounded px-3 py-2.5 text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent/50"
              />
            </div>
          )}

          {mode === 'login' && !isDemo && (
            <button
              type="button"
              onClick={handleForgotPassword}
              className="text-xs text-accent hover:underline"
            >
              {t('login.forgotPassword')}
            </button>
          )}

          {error && <p className="text-sm text-danger" role="alert">{error}</p>}
          {info && <p className="text-sm text-accent" role="status">{info}</p>}

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
              <button onClick={() => { setMode('register'); setError(''); setInfo('') }} className="text-accent hover:underline">
                {t('login.createOne')}
              </button>
            </>
          ) : (
            <>
              {t('login.hasAccount')}{' '}
              <button onClick={() => { setMode('login'); setError(''); setInfo('') }} className="text-accent hover:underline">
                {t('login.signInLink')}
              </button>
            </>
          )}
        </p>
      </div>
    </div>
  )
}
