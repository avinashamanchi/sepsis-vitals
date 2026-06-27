import { useEffect, useState, useCallback } from 'react'
import { useTranslation } from 'react-i18next'
import { Clock } from 'lucide-react'
import { useStore } from '../stores/useStore'
import { api, isDemo } from '../lib/api'

export function SessionWarning() {
  const { t } = useTranslation()
  const show = useStore((s) => s.showSessionWarning)
  const setShow = useStore((s) => s.setShowSessionWarning)
  const updateActivity = useStore((s) => s.updateActivity)
  const token = useStore((s) => s.token)
  const [secondsLeft, setSecondsLeft] = useState(60)

  // Reset countdown when modal opens
  useEffect(() => {
    if (show) setSecondsLeft(60)
  }, [show])

  // Countdown timer
  useEffect(() => {
    if (!show) return
    const interval = setInterval(() => {
      setSecondsLeft((prev) => {
        if (prev <= 1) {
          clearInterval(interval)
          return 0
        }
        return prev - 1
      })
    }, 1000)
    return () => clearInterval(interval)
  }, [show])

  const handleKeepWorking = useCallback(async () => {
    setShow(false)
    updateActivity()
    if (!isDemo) {
      try {
        await api.ping()
      } catch {
        // If ping fails, the next real request will handle re-auth
      }
    }
  }, [setShow, updateActivity])

  if (!show || !token) return null

  const progress = (secondsLeft / 60) * 100

  return (
    <div className="fixed inset-0 z-[200] flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="bg-surface border border-border rounded-lg shadow-2xl w-full max-w-sm mx-4 p-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 rounded-full bg-warning/20">
            <Clock className="w-6 h-6 text-warning" />
          </div>
          <h2 className="font-heading text-lg font-semibold text-text-primary">
            {t('common.sessionExpiring', { seconds: '' }).split('{{')[0].trim() || 'Session Expiring'}
          </h2>
        </div>

        <p className="text-sm text-text-secondary mb-4">
          {t('common.sessionExpiring', { seconds: secondsLeft })}
        </p>

        {/* Progress bar */}
        <div className="w-full h-2 bg-border rounded-full mb-6 overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-1000 ease-linear"
            style={{
              width: `${progress}%`,
              backgroundColor: secondsLeft > 30 ? 'var(--color-warning)' : 'var(--color-danger)',
            }}
          />
        </div>

        <button
          onClick={handleKeepWorking}
          className="w-full py-2.5 rounded-md bg-accent text-background font-semibold text-sm hover:bg-accent/90 transition-colors focus:outline-none focus:ring-2 focus:ring-accent/50"
        >
          {t('common.keepWorking')}
        </button>
      </div>
    </div>
  )
}
