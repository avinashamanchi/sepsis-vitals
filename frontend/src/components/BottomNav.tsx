import { NavLink } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import {
  Activity, Bell, Brain, Globe,
  LayoutDashboard, Users,
} from 'lucide-react'
import clsx from 'clsx'

const NAV_ITEMS = [
  { to: '/dashboard', icon: LayoutDashboard, key: 'nav.home' },
  { to: '/patients', icon: Users, key: 'nav.patients' },
  { to: '/monitor', icon: Activity, key: 'nav.monitor' },
  { to: '/predict', icon: Brain, key: 'nav.predict' },
  { to: '/population', icon: Globe, key: 'nav.population' },
  { to: '/alerts', icon: Bell, key: 'nav.alerts' },
]

export function BottomNav() {
  const { t } = useTranslation()

  return (
    <nav aria-label={t('nav.mobileNav')} className="fixed bottom-0 left-0 right-0 z-50 lg:hidden bg-surface/95 backdrop-blur-md border-t border-border">
      <div className="flex items-center justify-around h-14">
        {NAV_ITEMS.map(({ to, icon: Icon, key }) => (
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
            <Icon className="w-5 h-5" aria-hidden="true" />
            {t(key)}
          </NavLink>
        ))}
      </div>
    </nav>
  )
}
