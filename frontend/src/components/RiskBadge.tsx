import clsx from 'clsx'
import { ShieldCheck, AlertTriangle, AlertCircle, AlertOctagon } from 'lucide-react'
import type { RiskLevel } from '../types'
import { RISK_COLORS, RISK_BG, riskLabel } from '../lib/risk'

const RISK_ICONS: Record<RiskLevel, typeof ShieldCheck> = {
  low: ShieldCheck,
  moderate: AlertTriangle,
  high: AlertCircle,
  critical: AlertOctagon,
}

interface RiskBadgeProps {
  level: RiskLevel
  size?: 'sm' | 'md'
  pulse?: boolean
}

export function RiskBadge({ level, size = 'sm', pulse = false }: RiskBadgeProps) {
  const Icon = RISK_ICONS[level]
  return (
    <span
      className={clsx(
        'inline-flex items-center font-medium rounded-full',
        RISK_COLORS[level],
        RISK_BG[level],
        size === 'sm' ? 'text-xs px-2 py-0.5' : 'text-sm px-3 py-1',
        pulse && level === 'critical' && 'animate-pulse-critical',
      )}
      role="status"
      aria-label={`Risk level: ${riskLabel(level)}`}
    >
      <Icon className={clsx(size === 'sm' ? 'w-3 h-3' : 'w-3.5 h-3.5', 'mr-1')} aria-hidden="true" />
      {riskLabel(level)}
    </span>
  )
}
