import { Activity, ArrowUpRight } from 'lucide-react'
import { Link, NavLink } from 'react-router-dom'

const navItems = [
  { to: '/evidence', label: 'Evidence' },
  { to: '/pilot', label: 'Pilot program' },
]

export function PublicHeader() {
  return (
    <header className="sticky top-0 z-50 border-b border-border bg-void/92 backdrop-blur-xl">
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-5 sm:px-8">
        <Link
          to="/"
          className="flex items-center gap-2.5 font-heading text-sm font-bold tracking-tight text-text-primary"
          aria-label="Sepsis Vitals home"
        >
          <span className="grid h-8 w-8 place-items-center rounded-lg border border-accent/25 bg-accent/10">
            <Activity className="h-4 w-4 text-accent" aria-hidden="true" />
          </span>
          <span>Sepsis Vitals</span>
        </Link>

        <nav className="flex items-center gap-1" aria-label="Public navigation">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                `hidden rounded-md px-3 py-2 text-xs transition-colors sm:block ${
                  isActive
                    ? 'bg-elevated text-text-primary'
                    : 'text-text-secondary hover:text-text-primary'
                }`
              }
            >
              {item.label}
            </NavLink>
          ))}
          <Link
            to="/login"
            className="ml-2 inline-flex items-center gap-1.5 rounded-md border border-accent/35 bg-accent/10 px-3.5 py-2 text-xs font-semibold text-accent transition-colors hover:bg-accent/15"
          >
            Research demo
            <ArrowUpRight className="h-3.5 w-3.5" aria-hidden="true" />
          </Link>
        </nav>
      </div>
    </header>
  )
}

export function PublicFooter() {
  return (
    <footer className="border-t border-border bg-void px-5 py-10 sm:px-8">
      <div className="mx-auto flex max-w-6xl flex-col gap-6 text-xs text-text-muted sm:flex-row sm:items-end sm:justify-between">
        <div>
          <div className="mb-2 flex items-center gap-2 font-heading font-semibold text-text-secondary">
            <Activity className="h-4 w-4 text-accent" aria-hidden="true" />
            Sepsis Vitals
          </div>
          <p className="max-w-lg leading-relaxed">
            Investigational software for retrospective research and silent-mode evaluation.
            Not cleared for diagnosis, treatment, or autonomous clinical use.
          </p>
        </div>
        <div className="flex gap-5">
          <Link to="/evidence" className="hover:text-text-primary">Evidence</Link>
          <Link to="/pilot" className="hover:text-text-primary">Pilot</Link>
          <a href="mailto:sales@sepsisvitals.com" className="hover:text-text-primary">Contact</a>
        </div>
      </div>
    </footer>
  )
}

export function PublicShell({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-void text-text-primary">
      <PublicHeader />
      <main>{children}</main>
      <PublicFooter />
    </div>
  )
}
