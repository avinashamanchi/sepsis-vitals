import clsx from 'clsx'
import type { RiskLevel } from '../types'
import { RISK_COLORS, RISK_BG, riskLabel } from '../lib/risk'

interface RiskBadgeProps {
  level: RiskLevel
  size?: 'sm' | 'md'
  pulse?: boolean
}

export function RiskBadge({ level, size = 'sm', pulse = false }: RiskBadgeProps) {
  return (
    <span
      className={clsx(
        'inline-flex items-center font-medium rounded-full',
        RISK_COLORS[level],
        RISK_BG[level],
        size === 'sm' ? 'text-xs px-2 py-0.5' : 'text-sm px-3 py-1',
        pulse && level === 'critical' && 'animate-pulse-critical',
      )}
    >
      <span
        className={clsx(
          'w-1.5 h-1.5 rounded-full mr-1.5',
          level === 'low' && 'bg-risk-low',
          level === 'moderate' && 'bg-risk-moderate',
          level === 'high' && 'bg-risk-high',
          level === 'critical' && 'bg-risk-critical',
        )}
      />
      {riskLabel(level)}
    </span>
  )
}
