import { useLocation } from 'react-router-dom'
import { Menu, Bell } from 'lucide-react'
import { useStore } from '../stores/useStore'

const PAGE_TITLES: Record<string, string> = {
  '/': 'Dashboard',
  '/patients': 'Patients',
  '/scores': 'Score Lab',
  '/predict': 'AI Predict',
  '/analytics': 'Analytics',
  '/alerts': 'Alerts',
  '/admin': 'Admin',
}

export function TopBar() {
  const { setSidebarOpen, alerts } = useStore()
  const unread = alerts.filter((a) => !a.dismissed).length
  const { pathname } = useLocation()
  const pageTitle = PAGE_TITLES[pathname] || 'Sepsis Vitals'

  return (
    <header className="sticky top-0 z-30 bg-surface/80 backdrop-blur-md border-b border-border px-4 lg:px-6 h-14 flex items-center justify-between">
      <div className="flex items-center gap-3">
        <button
          onClick={() => setSidebarOpen(true)}
          className="lg:hidden p-2 -ml-2 text-text-secondary hover:text-text-primary"
        >
          <Menu className="w-5 h-5" />
        </button>
        <span className="lg:hidden font-heading text-sm font-semibold text-text-primary">
          {pageTitle}
        </span>
      </div>

      <div className="flex-1" />

      <div className="flex items-center gap-4">
        <button className="relative p-2 text-text-secondary hover:text-text-primary transition-colors">
          <Bell className="w-5 h-5" />
          {unread > 0 && (
            <span className="absolute -top-0.5 -right-0.5 w-4 h-4 rounded-full bg-danger text-[10px] font-bold flex items-center justify-center text-white">
              {unread > 9 ? '9+' : unread}
            </span>
          )}
        </button>
      </div>
    </header>
  )
}
