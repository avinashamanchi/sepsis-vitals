import { NavLink } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { useStore } from '../stores/useStore'
import {
  Activity, BarChart3, Bell, Brain, Calculator,
  LayoutDashboard, LogOut, Settings, Shield, Users, Wifi, WifiOff,
} from 'lucide-react'
import clsx from 'clsx'

const NAV_ITEMS = [
  { to: '/', icon: LayoutDashboard, key: 'nav.dashboard' },
  { to: '/patients', icon: Users, key: 'nav.patients' },
  { to: '/monitor', icon: Activity, key: 'nav.monitor' },
  { to: '/scores', icon: Calculator, key: 'nav.scoreLab' },
  { to: '/predict', icon: Brain, key: 'nav.predict' },
  { to: '/analytics', icon: BarChart3, key: 'nav.analytics' },
  { to: '/alerts', icon: Bell, key: 'nav.alerts' },
  { to: '/admin', icon: Settings, key: 'nav.admin' },
]

export function Sidebar() {
  const { t } = useTranslation()
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
          aria-label={t('nav.closeNav')}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => { if (e.key === 'Escape') setSidebarOpen(false) }}
        />
      )}

      <aside
        aria-label={t('nav.mainNav')}
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
            {t('app.title')}
          </h1>
          <div className="flex items-center gap-1.5 mt-2 text-xs text-text-muted">
            {wsConnected ? (
              <><Wifi className="w-3 h-3 text-accent" /> {t('common.live')}</>
            ) : (
              <><WifiOff className="w-3 h-3 text-danger" /> {t('common.offline')}</>
            )}
          </div>
        </div>

        {/* Navigation */}
        <nav aria-label={t('nav.siteNav')} className="flex-1 py-3 overflow-y-auto">
          {NAV_ITEMS.map(({ to, icon: Icon, key }) => (
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
              {t(key)}
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
            {t('nav.signOut')}
          </button>
        </div>
      </aside>
    </>
  )
}
