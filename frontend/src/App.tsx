import { lazy, Suspense, useEffect, useCallback } from 'react'
import { Routes, Route, Navigate, useLocation, useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { FlaskConical } from 'lucide-react'
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

const Landing = lazy(() => import('./pages/Landing').then((m) => ({ default: m.Landing })))
const Evidence = lazy(() => import('./pages/Evidence').then((m) => ({ default: m.Evidence })))
const Pilot = lazy(() => import('./pages/Pilot').then((m) => ({ default: m.Pilot })))
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

const SESSION_TIMEOUT_MS = 15 * 60 * 1000 // shared-workstation safety timeout

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

  if (!token) {
    return <Navigate to="/login" state={{ from: location }} replace />
  }

  return <>{children}</>
}

/** Redirect `/` based on auth state: authenticated → /dashboard, else → Landing */
function RootRedirect() {
  const token = useStore((s) => s.token)
  if (token) {
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
  useWebSocket()

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

  return (
    <Suspense fallback={<PageLoading />}>
      <Routes>
        {/* Public routes stay public. Legal acceptance belongs at the app boundary. */}
        <Route path="/" element={<RootRedirect />} />
        <Route path="/evidence" element={<Evidence />} />
        <Route path="/pilot" element={<Pilot />} />
        <Route path="/pricing" element={<Navigate to="/pilot" replace />} />
        <Route path="/login" element={<Login />} />

        <Route
          path="/*"
          element={
            <EulaGate>
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
                    <div
                      role="note"
                      className="flex items-start gap-2 border-b border-warning/20 bg-warning/8 px-4 py-2.5 text-[11px] leading-relaxed text-warning lg:px-6"
                    >
                      <FlaskConical className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                      <span>
                        <strong>Research environment.</strong> The model is trained on synthetic data
                        and has not been clinically validated. Do not use outputs for diagnosis or treatment.
                      </span>
                    </div>
                    <main id="main-content" className="flex-1 p-4 lg:p-6 pb-20 lg:pb-6">
                      <Suspense fallback={<PageLoading />}>
                        <Routes>
                          <Route path="/dashboard" element={<Dashboard />} />
                          <Route path="/patients" element={<Patients />} />
                          <Route path="/patients/:id" element={<PatientDetail />} />
                          <Route path="/monitor" element={<Monitor />} />
                          <Route path="/scores" element={<ScoreLab />} />
                          <Route path="/predict" element={<Predict />} />
                          <Route path="/analytics" element={<Analytics />} />
                          <Route path="/alerts" element={<Alerts />} />
                          <Route path="/admin" element={<Admin />} />
                          <Route path="*" element={<Navigate to="/dashboard" replace />} />
                        </Routes>
                      </Suspense>
                    </main>
                  </div>
                  <SessionWarning />
                  <BottomNav />
                  <SimulatorPanel />
                </div>
              </AuthGuard>
            </EulaGate>
          }
        />
      </Routes>
    </Suspense>
  )
}
