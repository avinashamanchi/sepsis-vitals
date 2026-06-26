import { lazy, Suspense, useEffect, useCallback } from 'react'
import { Routes, Route, Navigate, useLocation, useNavigate } from 'react-router-dom'
import { EulaGate } from './components/EulaGate'
import { Sidebar } from './components/Sidebar'
import { TopBar } from './components/TopBar'
import { BottomNav } from './components/BottomNav'
import { SimulatorPanel } from './components/SimulatorPanel'
import { useWebSocket } from './hooks/useWebSocket'
import { useStore } from './stores/useStore'
import { isDemo, setOnUnauthorized, api } from './lib/api'

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
  const location = useLocation()
  const navigate = useNavigate()

  // Wire up 401 handler
  useEffect(() => {
    setOnUnauthorized(() => {
      logout()
      navigate('/login')
    })
  }, [logout, navigate])

  // Session timeout check
  useEffect(() => {
    if (!token || isDemo) return
    const interval = setInterval(() => {
      if (Date.now() - lastActivity > SESSION_TIMEOUT_MS) {
        logout()
        navigate('/login')
      }
    }, 30_000) // check every 30s
    return () => clearInterval(interval)
  }, [token, lastActivity, logout, navigate])

  // Track activity on route change
  useEffect(() => {
    if (token) updateActivity()
  }, [location.pathname, token, updateActivity])

  // Track activity on user interaction
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

  // Redirect to login if not authenticated (skip in demo mode)
  if (!token && !isDemo) {
    return <Navigate to="/login" state={{ from: location }} replace />
  }

  return <>{children}</>
}

export default function App() {
  useWebSocket()

  useEffect(() => {
    if (isDemo) return
    api.simulatorSessions()
      .then(() => useStore.getState().setSimulatorEnabled(true))
      .catch(() => useStore.getState().setSimulatorEnabled(false))
  }, [])

  return (
    <EulaGate>
      <Suspense fallback={<PageLoading />}>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route
            path="/*"
            element={
              <AuthGuard>
                <div className="min-h-screen bg-background text-text-primary font-mono">
                  {/* Skip to content link for a11y */}
                  <a
                    href="#main-content"
                    className="sr-only focus:not-sr-only focus:fixed focus:top-2 focus:left-2 focus:z-[100] focus:bg-accent focus:text-background focus:px-4 focus:py-2 focus:rounded"
                  >
                    Skip to content
                  </a>
                  <Sidebar />
                  <div className="lg:ml-[220px] min-h-screen flex flex-col">
                    <TopBar />
                    <main id="main-content" className="flex-1 p-4 lg:p-6 pb-20 lg:pb-6">
                      <Suspense fallback={<PageLoading />}>
                        <Routes>
                          <Route path="/" element={<Dashboard />} />
                          <Route path="/patients" element={<Patients />} />
                          <Route path="/patients/:id" element={<PatientDetail />} />
                          <Route path="/monitor" element={<Monitor />} />
                          <Route path="/scores" element={<ScoreLab />} />
                          <Route path="/predict" element={<Predict />} />
                          <Route path="/analytics" element={<Analytics />} />
                          <Route path="/alerts" element={<Alerts />} />
                          <Route path="/admin" element={<Admin />} />
                        </Routes>
                      </Suspense>
                    </main>
                  </div>
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
