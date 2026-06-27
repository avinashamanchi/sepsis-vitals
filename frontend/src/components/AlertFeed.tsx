import { useState } from 'react'
import { X, Check, MessageSquare } from 'lucide-react'
import clsx from 'clsx'
import { useStore } from '../stores/useStore'
import { RISK_BORDER } from '../lib/risk'
import { RiskBadge } from './RiskBadge'
import { useTranslation } from 'react-i18next'

export function AlertFeed({ limit = 5 }: { limit?: number }) {
  const { t } = useTranslation()
  const alerts = useStore((s) => s.alerts)
  const dismissAlert = useStore((s) => s.dismissAlert)
  const acknowledgeAlert = useStore((s) => s.acknowledgeAlert)
  const setAlertNote = useStore((s) => s.setAlertNote)
  const visible = alerts.filter((a) => !a.dismissed).slice(0, limit)
  const [expandedId, setExpandedId] = useState<string | null>(null)
  const [noteText, setNoteText] = useState('')

  if (visible.length === 0) {
    return (
      <div className="text-center py-8 text-text-muted text-sm">
        {t('alertFeed.noAlerts')}
      </div>
    )
  }

  return (
    <div className="space-y-2" role="log" aria-live="polite" aria-label={t('alertFeed.feedLabel')}>
      {visible.map((alert) => (
        <div
          key={alert.id}
          className={clsx(
            'p-3 bg-surface border-l-3 rounded-r-lg animate-fade-in',
            RISK_BORDER[alert.riskLevel],
            alert.riskLevel === 'critical' && 'glow-danger',
          )}
          role="alert"
        >
          <div className="flex items-start gap-3">
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-1">
                <RiskBadge level={alert.riskLevel} pulse={alert.riskLevel === 'critical'} />
                {alert.acknowledged && (
                  <span className="text-xs text-accent flex items-center gap-1">
                    <Check className="w-3 h-3" /> {t('alertFeed.ack')}
                  </span>
                )}
                <span className="text-xs text-text-muted">
                  {new Date(alert.timestamp).toLocaleTimeString()}
                </span>
              </div>
              <p className="text-sm text-text-secondary">{alert.message}</p>
              {alert.note && (
                <p className="text-xs text-text-muted mt-1 italic">Note: {alert.note}</p>
              )}
            </div>
            <div className="flex items-center gap-1 shrink-0">
              {!alert.acknowledged && (
                <button
                  onClick={() => acknowledgeAlert(alert.id)}
                  className="p-1 text-text-muted hover:text-accent transition-colors"
                  aria-label={t('alertFeed.acknowledgeLabel')}
                  title={t('alertFeed.acknowledge')}
                >
                  <Check className="w-3.5 h-3.5" />
                </button>
              )}
              <button
                onClick={() => {
                  setExpandedId(expandedId === alert.id ? null : alert.id)
                  setNoteText(alert.note ?? '')
                }}
                className="p-1 text-text-muted hover:text-info transition-colors"
                aria-label={t('alertFeed.addNoteLabel')}
                title={t('alertFeed.addNote')}
              >
                <MessageSquare className="w-3.5 h-3.5" />
              </button>
              <button
                onClick={() => dismissAlert(alert.id)}
                className="p-1 text-text-muted hover:text-text-primary transition-colors"
                aria-label={t('alertFeed.dismissLabel')}
                title={t('alertFeed.dismiss')}
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
          {/* Note input */}
          {expandedId === alert.id && (
            <div className="mt-2 flex gap-2">
              <input
                type="text"
                value={noteText}
                onChange={(e) => setNoteText(e.target.value)}
                placeholder={t('alertFeed.notePlaceholder')}
                className="flex-1 bg-elevated border border-border rounded px-2 py-1 text-xs text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent/50"
                aria-label={t('alertFeed.noteInputLabel')}
              />
              <button
                onClick={() => {
                  setAlertNote(alert.id, noteText)
                  setExpandedId(null)
                }}
                className="px-2 py-1 text-xs text-accent bg-accent/10 border border-accent/30 rounded hover:bg-accent/20"
              >
                {t('common.save')}
              </button>
            </div>
          )}
        </div>
      ))}
    </div>
  )
}
