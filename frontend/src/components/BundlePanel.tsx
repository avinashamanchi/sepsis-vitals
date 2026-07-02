import { useEffect, useMemo, useState } from 'react'
import clsx from 'clsx'
import { useTranslation } from 'react-i18next'
import {
  Check,
  Clock,
  Syringe,
  FlaskConical,
  Droplets,
  Activity,
  TestTube,
  AlertOctagon,
} from 'lucide-react'

/**
 * BundlePanel
 * -----------
 * Live Hour-1 Sepsis Bundle tracker. Renders the running clock against the
 * 60-minute antibiotic target, a tappable checklist of protocol tasks (each
 * stamped when completed), and the two headline KPIs — time-to-antibiotics and
 * bundle compliance. Backed by the /bundles API.
 *
 * The panel is intentionally self-contained: pass a `patientId` and an async
 * `client` implementing the four calls it needs. That keeps it decoupled from
 * the app's api singleton and trivially storybook-able.
 */

export interface BundleTaskView {
  task_key: string
  label: string
  conditional: string | null
  depends_on: string | null
  order: number
  target_minutes: number
  critical: boolean
  completed: boolean
  completed_at: string | null
  minutes_from_start: number | null
  overdue: boolean
  note: string | null
}

export interface BundleView {
  id: string
  patient_id: string
  status: 'open' | 'completed' | 'expired' | 'cancelled'
  risk_level_at_start: string | null
  started_at: string | null
  elapsed_seconds: number
  seconds_remaining: number | null
  time_to_antibiotics_s: number | null
  compliance_pct: number | null
  tasks: BundleTaskView[]
}

export interface BundleClient {
  getForPatient: (patientId: string) => Promise<BundleView | null>
  start: (patientId: string) => Promise<BundleView>
  completeTask: (
    bundleId: string,
    taskKey: string,
    completed: boolean,
  ) => Promise<BundleView>
  cancel: (bundleId: string) => Promise<BundleView>
}

const TASK_ICONS: Record<string, typeof Syringe> = {
  lactate_initial: FlaskConical,
  blood_cultures: TestTube,
  antibiotics: Syringe,
  fluids: Droplets,
  vasopressors: Activity,
  lactate_repeat: FlaskConical,
}

function fmtClock(totalSeconds: number): string {
  const sign = totalSeconds < 0 ? '-' : ''
  const s = Math.abs(Math.round(totalSeconds))
  const m = Math.floor(s / 60)
  const sec = s % 60
  return `${sign}${m}:${sec.toString().padStart(2, '0')}`
}

interface BundlePanelProps {
  patientId: string
  client: BundleClient
  currentUserId?: string
}

export function BundlePanel({ patientId, client }: BundlePanelProps) {
  const { t } = useTranslation()
  const [bundle, setBundle] = useState<BundleView | null>(null)
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [tick, setTick] = useState(0)

  // Initial load.
  useEffect(() => {
    let live = true
    setLoading(true)
    client
      .getForPatient(patientId)
      .then((b) => live && setBundle(b))
      .catch(() => live && setBundle(null))
      .finally(() => live && setLoading(false))
    return () => {
      live = false
    }
  }, [patientId, client])

  // 1 Hz clock tick while a bundle is open.
  useEffect(() => {
    if (bundle?.status !== 'open') return
    const id = window.setInterval(() => setTick((x) => x + 1), 1000)
    return () => window.clearInterval(id)
  }, [bundle?.status])

  // Derive live remaining seconds from the last server value + local drift.
  const liveRemaining = useMemo(() => {
    if (!bundle || bundle.seconds_remaining == null) return null
    return bundle.seconds_remaining - tick
  }, [bundle, tick])

  async function handleStart() {
    setBusy('start')
    setError(null)
    try {
      setBundle(await client.start(patientId))
      setTick(0)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to start bundle')
    } finally {
      setBusy(null)
    }
  }

  async function handleToggle(task: BundleTaskView) {
    if (!bundle) return
    setBusy(task.task_key)
    setError(null)
    try {
      const updated = await client.completeTask(
        bundle.id,
        task.task_key,
        !task.completed,
      )
      setBundle(updated)
      setTick(0)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to update task')
    } finally {
      setBusy(null)
    }
  }

  if (loading) {
    return (
      <div className="bg-surface border border-border rounded-lg p-4 text-sm text-text-muted">
        {t('common.loading', 'Loading…')}
      </div>
    )
  }

  // No open bundle → offer to start one.
  if (!bundle || bundle.status !== 'open') {
    const closed = bundle && bundle.status !== 'open'
    return (
      <div className="bg-surface border border-border rounded-lg p-4">
        <div className="flex items-center justify-between mb-2">
          <h3 className="font-heading font-semibold text-text-primary">
            {t('bundle.title', 'Hour-1 Sepsis Bundle')}
          </h3>
          {closed && (
            <span className="text-xs text-text-muted uppercase tracking-wider">
              {t(`bundle.status.${bundle!.status}`, bundle!.status)}
            </span>
          )}
        </div>
        {closed && bundle!.time_to_antibiotics_s != null && (
          <p className="text-xs text-text-secondary mb-3">
            {t('bundle.abxTime', 'Antibiotics given at')}{' '}
            {fmtClock(bundle!.time_to_antibiotics_s)} ·{' '}
            {bundle!.compliance_pct ?? 0}% {t('bundle.compliant', 'compliant')}
          </p>
        )}
        <button
          onClick={handleStart}
          disabled={busy === 'start'}
          className="w-full bg-danger/90 hover:bg-danger text-white font-medium rounded-md px-4 py-2 text-sm disabled:opacity-60"
        >
          {busy === 'start'
            ? t('common.loading', 'Loading…')
            : t('bundle.start', 'Start Hour-1 bundle')}
        </button>
        {error && <p className="text-xs text-danger mt-2">{error}</p>}
      </div>
    )
  }

  const remaining = liveRemaining ?? 0
  const isOverdue = remaining < 0
  const doneCritical = bundle.tasks.filter((x) => x.critical && x.completed).length
  const totalCritical = bundle.tasks.filter((x) => x.critical).length

  return (
    <div className="bg-surface border border-border rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-heading font-semibold text-text-primary flex items-center gap-2">
          <Clock className="w-4 h-4" aria-hidden="true" />
          {t('bundle.title', 'Hour-1 Sepsis Bundle')}
        </h3>
        <button
          onClick={() => client.cancel(bundle.id).then(setBundle).catch(() => {})}
          className="text-xs text-text-muted hover:text-text-secondary"
        >
          {t('bundle.close', 'Close')}
        </button>
      </div>

      {/* Countdown + KPIs */}
      <div className="grid grid-cols-3 gap-2 mb-4">
        <div
          className={clsx(
            'rounded-md p-3 text-center',
            isOverdue ? 'bg-danger/10' : 'bg-bg',
          )}
          aria-live="polite"
        >
          <p className="text-[10px] uppercase tracking-wider text-text-muted">
            {isOverdue
              ? t('bundle.overdueBy', 'Overdue by')
              : t('bundle.timeLeft', 'Time to abx')}
          </p>
          <p
            className={clsx(
              'text-xl font-bold font-heading tabular-nums',
              isOverdue ? 'text-danger animate-pulse-critical' : 'text-text-primary',
            )}
          >
            {fmtClock(remaining)}
          </p>
        </div>
        <div className="rounded-md p-3 text-center bg-bg">
          <p className="text-[10px] uppercase tracking-wider text-text-muted">
            {t('bundle.tasksDone', 'Tasks')}
          </p>
          <p className="text-xl font-bold font-heading text-text-primary tabular-nums">
            {doneCritical}/{totalCritical}
          </p>
        </div>
        <div className="rounded-md p-3 text-center bg-bg">
          <p className="text-[10px] uppercase tracking-wider text-text-muted">
            {t('bundle.compliance', 'Compliance')}
          </p>
          <p className="text-xl font-bold font-heading text-accent tabular-nums">
            {bundle.compliance_pct ?? 0}%
          </p>
        </div>
      </div>

      {/* Task checklist */}
      <ul className="space-y-2">
        {bundle.tasks.map((task) => {
          const Icon = TASK_ICONS[task.task_key] ?? Activity
          const isBusy = busy === task.task_key
          return (
            <li key={task.task_key}>
              <button
                onClick={() => handleToggle(task)}
                disabled={isBusy}
                className={clsx(
                  'w-full flex items-start gap-3 rounded-md border px-3 py-2 text-left transition-colors',
                  task.completed
                    ? 'border-accent/40 bg-accent/5'
                    : task.overdue
                      ? 'border-danger/40 bg-danger/5'
                      : 'border-border bg-bg hover:border-text-muted',
                  isBusy && 'opacity-60',
                )}
              >
                <span
                  className={clsx(
                    'mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full border',
                    task.completed
                      ? 'bg-accent border-accent text-white'
                      : 'border-text-muted text-transparent',
                  )}
                  aria-hidden="true"
                >
                  <Check className="h-3 w-3" />
                </span>
                <span className="flex-1 min-w-0">
                  <span className="flex items-center gap-1.5">
                    <Icon className="w-3.5 h-3.5 text-text-muted" aria-hidden="true" />
                    <span
                      className={clsx(
                        'text-sm font-medium',
                        task.completed
                          ? 'text-text-secondary line-through'
                          : 'text-text-primary',
                      )}
                    >
                      {t(`bundle.task.${task.task_key}`, task.label)}
                    </span>
                    {!task.critical && (
                      <span className="text-[10px] text-text-muted">
                        {t('bundle.conditional', 'conditional')}
                      </span>
                    )}
                  </span>
                  {task.conditional && !task.completed && (
                    <span className="block text-[11px] text-text-muted mt-0.5">
                      {task.conditional}
                    </span>
                  )}
                  {task.completed && task.minutes_from_start != null && (
                    <span className="block text-[11px] text-accent mt-0.5">
                      {t('bundle.at', 'at')} {task.minutes_from_start.toFixed(0)}{' '}
                      {t('bundle.min', 'min')}
                    </span>
                  )}
                </span>
                {task.overdue && !task.completed && (
                  <AlertOctagon
                    className="w-4 h-4 text-danger shrink-0"
                    aria-label={t('bundle.overdue', 'Overdue')}
                  />
                )}
              </button>
            </li>
          )
        })}
      </ul>

      {error && <p className="text-xs text-danger mt-2">{error}</p>}
    </div>
  )
}
