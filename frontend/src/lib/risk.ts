import type { RiskLevel } from '../types'

export const RISK_COLORS: Record<RiskLevel, string> = {
  low: 'text-risk-low',
  moderate: 'text-risk-moderate',
  high: 'text-risk-high',
  critical: 'text-risk-critical',
}

export const RISK_BG: Record<RiskLevel, string> = {
  low: 'bg-risk-low/10',
  moderate: 'bg-risk-moderate/10',
  high: 'bg-risk-high/10',
  critical: 'bg-risk-critical/10',
}

export const RISK_BORDER: Record<RiskLevel, string> = {
  low: 'border-risk-low',
  moderate: 'border-risk-moderate',
  high: 'border-risk-high',
  critical: 'border-risk-critical',
}

export function riskLabel(level: RiskLevel): string {
  return level.charAt(0).toUpperCase() + level.slice(1)
}

export function probabilityToRisk(p: number): RiskLevel {
  if (p >= 0.75) return 'critical'
  if (p >= 0.50) return 'high'
  if (p >= 0.25) return 'moderate'
  return 'low'
}
