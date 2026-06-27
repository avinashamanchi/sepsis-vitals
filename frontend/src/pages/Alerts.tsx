import { useTranslation } from 'react-i18next'
import { AlertFeed } from '../components/AlertFeed'
import { useStore } from '../stores/useStore'
import { Activity, Trash2 } from 'lucide-react'

export function Alerts() {
  const { t } = useTranslation()
  const alerts = useStore((s) => s.alerts)
  const clearAlerts = useStore((s) => s.clearAlerts)
  const active = alerts.filter((a) => !a.dismissed).length

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="font-heading text-2xl font-bold flex items-center gap-2">
            <Activity className="w-6 h-6 text-danger" />
            {t('alerts.title')}
          </h1>
          <p className="text-sm text-text-secondary mt-1">
            {t('alerts.subtitle', { active, total: alerts.length })}
          </p>
        </div>
        {alerts.length > 0 && (
          <button
            onClick={clearAlerts}
            className="flex items-center gap-2 px-3 py-2 text-xs text-text-muted hover:text-danger border border-border rounded-lg transition-colors"
            aria-label={t('alerts.clearAllLabel')}
          >
            <Trash2 className="w-3.5 h-3.5" />
            {t('alerts.clearAll')}
          </button>
        )}
      </div>

      <div className="bg-surface border border-border rounded-lg p-4">
        <AlertFeed limit={50} />
      </div>
    </div>
  )
}
