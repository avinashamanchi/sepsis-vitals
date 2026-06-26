import { NavLink } from 'react-router-dom'
import { useStore } from '../stores/useStore'
import {
  Activity, BarChart3, Bell, Brain, Calculator,
  LayoutDashboard, LogOut, Settings, Shield, Users, Wifi, WifiOff,
} from 'lucide-react'
import clsx from 'clsx'

const NAV_ITEMS = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/patients', icon: Users, label: 'Patients' },
  { to: '/monitor', icon: Activity, label: 'Monitor' },
  { to: '/scores', icon: Calculator, label: 'Score Lab' },
  { to: '/predict', icon: Brain, label: 'AI Predict' },
  { to: '/analytics', icon: BarChart3, label: 'Analytics' },
  { to: '/alerts', icon: Bell, label: 'Alerts' },
  { to: '/admin', icon: Settings, label: 'Admin' },
]

export function Sidebar() {
  const wsConnected = useStore((s) => s.wsConnected)
  const logout = useStore((s) => s.logout)
  const sidebarOpen = useStore((s) => s.sidebarOpen)
  const setSidebarOpen = useStore((s) => s.setSidebarOpen)

  return (
    <>
      {/* Mobile overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          onClick={() => setSidebarOpen(false)}
          aria-label="Close navigation"
          role="button"
          tabIndex={0}
          onKeyDown={(e) => { if (e.key === 'Escape') setSidebarOpen(false) }}
        />
      )}

      <aside
        className={clsx(
          'fixed top-0 left-0 h-full w-[220px] bg-surface border-r border-border z-50',
          'flex flex-col transition-transform duration-200',
          'lg:translate-x-0',
          sidebarOpen ? 'translate-x-0' : '-translate-x-full',
        )}
      >
        {/* Logo */}
        <div className="p-5 border-b border-border">
          <h1 className="font-heading text-lg font-bold text-accent tracking-tight">
            <Shield className="inline-block w-5 h-5 mr-2 -mt-0.5" />
            Sepsis Vitals
          </h1>
          <div className="flex items-center gap-1.5 mt-2 text-xs text-text-muted">
            {wsConnected ? (
              <><Wifi className="w-3 h-3 text-accent" /> Live</>
            ) : (
              <><WifiOff className="w-3 h-3 text-danger" /> Offline</>
            )}
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 py-3 overflow-y-auto">
          {NAV_ITEMS.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              onClick={() => setSidebarOpen(false)}
              className={({ isActive }) =>
                clsx(
                  'flex items-center gap-3 px-5 py-2.5 text-sm transition-colors',
                  isActive
                    ? 'text-accent bg-accent/8 border-r-2 border-accent'
                    : 'text-text-secondary hover:text-text-primary hover:bg-elevated',
                )
              }
            >
              <Icon className="w-4 h-4" />
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Footer */}
        <div className="p-4 border-t border-border">
          <button
            onClick={logout}
            className="flex items-center gap-2 text-xs text-text-muted hover:text-danger transition-colors w-full"
          >
            <LogOut className="w-3.5 h-3.5" />
            Sign Out
          </button>
        </div>
      </aside>
    </>
  )
}
