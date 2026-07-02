import { useTranslation } from 'react-i18next'
import { useStore } from '../stores/useStore'
import { isDemo } from '../lib/api'

export function ConnectionStatus() {
  const { t } = useTranslation()
  const wsState = useStore((s) => s.wsState)
  const outboxPending = useStore((s) => s.outboxPending)

  // In demo mode, show a neutral offline state
  const state = isDemo ? 'offline' : wsState

  const config = {
    connected: {
      color: 'bg-emerald-500',
      pulse: true,
      label: t('common.connected'),
    },
    reconnecting: {
      color: 'bg-yellow-500',
      pulse: true,
      label: t('common.reconnecting', 'Reconnecting'),
    },
    offline: {
      color: 'bg-red-500',
      pulse: false,
      label: t('common.offline'),
    },
  }[state]

  return (
    <div className="flex items-center gap-1.5" title={config.label}>
      <span className="relative flex h-2.5 w-2.5">
        {config.pulse && (
          <span
            className={`absolute inline-flex h-full w-full rounded-full opacity-75 animate-ping ${config.color}`}
          />
        )}
        <span
          className={`relative inline-flex rounded-full h-2.5 w-2.5 ${config.color}`}
        />
      </span>
      <span className="hidden sm:inline text-[10px] text-text-muted font-mono">
        {config.label}
      </span>
      {outboxPending > 0 && (
        <span className="inline-flex items-center justify-center h-4 min-w-[16px] px-1 rounded-full bg-amber-500 text-[9px] font-bold text-white">
          {outboxPending}
        </span>
      )}
    </div>
  )
}
