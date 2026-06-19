import clsx from 'clsx'

interface StatCardProps {
  label: string
  value: string | number
  sublabel?: string
  color?: 'accent' | 'warning' | 'danger' | 'info' | 'default'
}

const COLOR_MAP = {
  accent: 'text-accent',
  warning: 'text-warning',
  danger: 'text-danger',
  info: 'text-info',
  default: 'text-text-primary',
}

export function StatCard({ label, value, sublabel, color = 'default' }: StatCardProps) {
  return (
    <div className="bg-surface border border-border rounded-lg p-4 animate-fade-in">
      <p className="text-xs text-text-muted uppercase tracking-wider mb-1">{label}</p>
      <p className={clsx('text-2xl font-bold font-heading', COLOR_MAP[color])}>
        {value}
      </p>
      {sublabel && (
        <p className="text-xs text-text-secondary mt-1">{sublabel}</p>
      )}
    </div>
  )
}
