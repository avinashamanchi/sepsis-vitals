import { lazy, Suspense, useEffect, useCallback, useState } from 'react'
import { Routes, Route, Navigate, useLocation, useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { LANGUAGES } from './i18n'
import { EulaGate } from './components/EulaGate'
import { Sidebar } from './components/Sidebar'
import { TopBar } from './components/TopBar'
import { BottomNav } from './components/BottomNav'
import { SimulatorPanel } from './components/SimulatorPanel'
import { SessionWarning } from './components/SessionWarning'
import { KeyboardShortcuts } from './components/KeyboardShortcuts'
import { useWebSocket } from './hooks/useWebSocket'
import { useStore } from './stores/useStore'
import { isDemo, setOnUnauthorized, api } from './lib/api'
import { onAuthChange } from './lib/auth'

const Landing = lazy(() => import('./pages/Landing').then((m) => ({ default: m.Landing })))
const Pricing = lazy(() => import('./pages/Pricing').then((m) => ({ default: m.Pricing })))
const Dashboard = lazy(() => import('./pages/Dashboard').then((m) => ({ default: m.Dashboard })))
const Patients = lazy(() => import('./pages/Patients').then((m) => ({ default: m.Patients })))
const PatientDetail = lazy(() => import('./pages/PatientDetail').then((m) => ({ default: m.PatientDetail })))
const ScoreLab = lazy(() => import('./pages/ScoreLab').then((m) => ({ default: m.ScoreLab })))
const Predict = lazy(() => import('./pages/Predict').then((m) => ({ default: m.Predict })))
const Analytics = lazy(() => import('./pages/Analytics').then((m) => ({ default: m.Analytics })))
const Alerts = lazy(() => import('./pages/Alerts').then((m) => ({ default: m.Alerts })))
const Admin = lazy(() => import('./pages/Admin').then((m) => ({ default: m.Admin })))
const Login = lazy(() => import('./pages/Login').then((m) => ({ default: m.Login })))
const Monitor = lazy(() => import('./pages/Monitor').then((m) => ({ default: m.Monitor })))
const Population = lazy(() => import('./pages/Population').then((m) => ({ default: m.Population })))

const SESSION_TIMEOUT_MS = 15 * 60 * 1000 // 15 minutes HIPAA

function PageLoading() {
  return (
    <div className="flex items-center justify-center py-20">
      <div className="w-6 h-6 border-2 border-accent border-t-transparent rounded-full animate-spin" />
    </div>
  )
}

function AuthGuard({ children }: { children: React.ReactNode }) {
  const token = useStore((s) => s.token)
  const lastActivity = useStore((s) => s.lastActivity)
  const logout = useStore((s) => s.logout)
  const updateActivity = useStore((s) => s.updateActivity)
  const setShowSessionWarning = useStore((s) => s.setShowSessionWarning)
  const location = useLocation()
  const navigate = useNavigate()

  useEffect(() => {
    setOnUnauthorized(() => {
      logout()
      navigate('/login')
    })
  }, [logout, navigate])

  useEffect(() => {
    if (!token || isDemo) return
    const interval = setInterval(() => {
      const idle = Date.now() - lastActivity
      if (idle > SESSION_TIMEOUT_MS) {
        setShowSessionWarning(false)
        logout()
        navigate('/login')
      } else if (idle > SESSION_TIMEOUT_MS - 60_000) {
        setShowSessionWarning(true)
      }
    }, 5_000)
    return () => clearInterval(interval)
  }, [token, lastActivity, logout, navigate, setShowSessionWarning])

  useEffect(() => {
    if (token) updateActivity()
  }, [location.pathname, token, updateActivity])

  const handleActivity = useCallback(() => {
    updateActivity()
  }, [updateActivity])

  useEffect(() => {
    if (!token) return
    window.addEventListener('click', handleActivity)
    window.addEventListener('keydown', handleActivity)
    return () => {
      window.removeEventListener('click', handleActivity)
      window.removeEventListener('keydown', handleActivity)
    }
  }, [token, handleActivity])

  if (!token && !isDemo) {
    return <Navigate to="/login" state={{ from: location }} replace />
  }

  return <>{children}</>
}

/** Redirect `/` based on auth state: authenticated → /dashboard, else → Landing */
function RootRedirect() {
  const token = useStore((s) => s.token)
  if (token || isDemo) {
    return <Navigate to="/dashboard" replace />
  }
  return (
    <Suspense fallback={<PageLoading />}>
      <Landing />
    </Suspense>
  )
}

export default function App() {
  const { t, i18n } = useTranslation()
  const setAuth = useStore((s) => s.setAuth)
  const [authReady, setAuthReady] = useState(isDemo)
  useWebSocket()

  // Firebase auth state listener — restore session on load
  useEffect(() => {
    if (isDemo) return
    const unsubscribe = onAuthChange(async (user) => {
      if (user) {
        const token = await user.getIdToken()
        setAuth(token, {
          email: user.email ?? '',
          role: 'user',
          displayName: user.displayName ?? undefined,
          photoURL: user.photoURL ?? undefined,
        })
      }
      setAuthReady(true)
    })
    return unsubscribe
  }, [setAuth])

  useEffect(() => {
    const lang = LANGUAGES.find((l) => l.code === i18n.language)
    const dir = lang?.dir ?? 'ltr'
    document.documentElement.dir = dir
    document.documentElement.lang = i18n.language
  }, [i18n.language])

  useEffect(() => {
    if (isDemo) return
    api.simulatorSessions()
      .then(() => useStore.getState().setSimulatorEnabled(true))
      .catch(() => useStore.getState().setSimulatorEnabled(false))
  }, [])

  if (!authReady) {
    return (
      <div className="min-h-screen bg-void flex items-center justify-center">
        <div className="w-6 h-6 border-2 border-accent border-t-transparent rounded-full animate-spin" />
      </div>
    )
  }

  return (
    <EulaGate>
      <Suspense fallback={<PageLoading />}>
        <Routes>
          {/* Public routes — no sidebar/topbar */}
          <Route path="/" element={<RootRedirect />} />
          <Route path="/pricing" element={<Pricing />} />
          <Route path="/login" element={<Login />} />

          {/* Authenticated routes — sidebar/topbar layout */}
          <Route
            path="/*"
            element={
              <AuthGuard>
                <div className="min-h-screen bg-background text-text-primary font-mono">
                  <a
                    href="#main-content"
                    className="sr-only focus:not-sr-only focus:fixed focus:top-2 focus:left-2 focus:z-[100] focus:bg-accent focus:text-background focus:px-4 focus:py-2 focus:rounded"
                  >
                    {t('app.skipToContent')}
                  </a>
                  <KeyboardShortcuts />
                  <Sidebar />
                  <div className="lg:ml-[220px] min-h-screen flex flex-col">
                    <TopBar />
                    <main id="main-content" className="flex-1 p-4 lg:p-6 pb-20 lg:pb-6">
                      <Suspense fallback={<PageLoading />}>
                        <Routes>
                          <Route path="/dashboard" element={<Dashboard />} />
                          <Route path="/patients" element={<Patients />} />
                          <Route path="/patients/:id" element={<PatientDetail />} />
                          <Route path="/monitor" element={<Monitor />} />
                          <Route path="/population" element={<Population />} />
                          <Route path="/scores" element={<ScoreLab />} />
                          <Route path="/predict" element={<Predict />} />
                          <Route path="/analytics" element={<Analytics />} />
                          <Route path="/alerts" element={<Alerts />} />
                          <Route path="/admin" element={<Admin />} />
                        </Routes>
                      </Suspense>
                    </main>
                  </div>
                  <SessionWarning />
                  <BottomNav />
                  <SimulatorPanel />
                </div>
              </AuthGuard>
            }
          />
        </Routes>
      </Suspense>
    </EulaGate>
  )
}
