import { X } from 'lucide-react'
import clsx from 'clsx'
import { useStore } from '../stores/useStore'
import { RISK_BORDER } from '../lib/risk'
import { RiskBadge } from './RiskBadge'

export function AlertFeed({ limit = 5 }: { limit?: number }) {
  const { alerts, dismissAlert } = useStore()
  const visible = alerts.filter((a) => !a.dismissed).slice(0, limit)

  if (visible.length === 0) {
    return (
      <div className="text-center py-8 text-text-muted text-sm">
        No active alerts
      </div>
    )
  }

  return (
    <div className="space-y-2">
      {visible.map((alert) => (
        <div
          key={alert.id}
          className={clsx(
            'flex items-start gap-3 p-3 bg-surface border-l-3 rounded-r-lg animate-fade-in',
            RISK_BORDER[alert.riskLevel],
            alert.riskLevel === 'critical' && 'glow-danger',
          )}
        >
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <RiskBadge level={alert.riskLevel} pulse={alert.riskLevel === 'critical'} />
              <span className="text-xs text-text-muted">
                {new Date(alert.timestamp).toLocaleTimeString()}
              </span>
            </div>
            <p className="text-sm text-text-secondary truncate">{alert.message}</p>
          </div>
          <button
            onClick={() => dismissAlert(alert.id)}
            className="p-1 text-text-muted hover:text-text-primary transition-colors shrink-0"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      ))}
    </div>
  )
}
