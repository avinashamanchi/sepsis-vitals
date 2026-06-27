import { NavLink } from 'react-router-dom'
import {
  Activity, Bell, Brain,
  LayoutDashboard, Settings, Users,
} from 'lucide-react'
import clsx from 'clsx'

const NAV_ITEMS = [
  { to: '/', icon: LayoutDashboard, label: 'Home' },
  { to: '/patients', icon: Users, label: 'Patients' },
  { to: '/monitor', icon: Activity, label: 'Monitor' },
  { to: '/predict', icon: Brain, label: 'Predict' },
  { to: '/alerts', icon: Bell, label: 'Alerts' },
  { to: '/admin', icon: Settings, label: 'Admin' },
]

export function BottomNav() {
  return (
    <nav className="fixed bottom-0 left-0 right-0 z-50 lg:hidden bg-surface/95 backdrop-blur-md border-t border-border">
      <div className="flex items-center justify-around h-14">
        {NAV_ITEMS.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              clsx(
                'flex flex-col items-center gap-0.5 px-2 py-1.5 text-[10px] transition-colors min-w-[48px]',
                isActive
                  ? 'text-accent'
                  : 'text-text-muted hover:text-text-secondary',
              )
            }
          >
            <Icon className="w-5 h-5" />
            {label}
          </NavLink>
        ))}
      </div>
    </nav>
  )
}
