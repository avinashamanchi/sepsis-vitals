import { useState, useEffect } from 'react'
import { Play, Square, ChevronDown, ChevronUp, FlaskConical } from 'lucide-react'
import { useStore } from '../stores/useStore'
import { api } from '../lib/api'
import clsx from 'clsx'
import { useTranslation } from 'react-i18next'

type Tab = 'replay' | 'ward'

interface MIMICCase {
  subject_id: number
  age_years: number
  sex: string
  sepsis_label: number
  icu_los_hours: number
  n_observations: number
}

export function SimulatorPanel() {
  const { t } = useTranslation()
  const simulatorEnabled = useStore((s) => s.simulatorEnabled)
  const sessions = useStore((s) => s.simulatorSessions)
  const addSimSession = useStore((s) => s.addSimSession)
  const removeSimSession = useStore((s) => s.removeSimSession)

  const [collapsed, setCollapsed] = useState(true)
  const [tab, setTab] = useState<Tab>('ward')
  const [cases, setCases] = useState<MIMICCase[]>([])
  const [loading, setLoading] = useState(false)

  // Ward sim controls
  const [wardPatients, setWardPatients] = useState(8)
  const [wardSpeed, setWardSpeed] = useState(360)
  const [wardSepsis, setWardSepsis] = useState(2)

  // Replay controls
  const [selectedCase, setSelectedCase] = useState<number | null>(null)
  const [replaySpeed, setReplaySpeed] = useState(720)

  // Load cases when panel opens
  useEffect(() => {
    if (!collapsed && simulatorEnabled && cases.length === 0) {
      api.simulatorCases()
        .then((data) => setCases(data.cases))
        .catch(() => {})
    }
  }, [collapsed, simulatorEnabled, cases.length])

  if (!simulatorEnabled) return null

  const startWard = async () => {
    setLoading(true)
    try {
      const res = await api.simulatorStartWard({
        n_patients: wardPatients,
        speed: wardSpeed,
        sepsis_count: wardSepsis,
      })
      addSimSession({
        session_id: res.session_id,
        type: 'ward',
        status: 'running',
        patient_count: wardPatients,
        started_at: Date.now() / 1000,
      })
    } catch { /* ignore */ }
    setLoading(false)
  }

  const startReplay = async () => {
    if (!selectedCase) return
    setLoading(true)
    try {
      const res = await api.simulatorStartReplay({
        subject_id: selectedCase,
        speed: replaySpeed,
      })
      addSimSession({
        session_id: res.session_id,
        type: 'replay',
        status: 'running',
        subject_id: res.subject_id,
        started_at: Date.now() / 1000,
      })
    } catch { /* ignore */ }
    setLoading(false)
  }

  const stopSession = async (sessionId: string) => {
    try {
      await api.simulatorStop(sessionId)
      removeSimSession(sessionId)
    } catch { /* ignore */ }
  }

  return (
    <div className="fixed bottom-4 right-4 z-50 w-80">
      {/* Toggle bar */}
      <button
        onClick={() => setCollapsed(!collapsed)}
        aria-expanded={!collapsed}
        className="w-full flex items-center justify-between bg-overlay border border-border rounded-t-lg px-4 py-2 text-sm font-medium text-text-primary hover:bg-elevated transition-colors"
      >
        <span className="flex items-center gap-2">
          <FlaskConical className="w-4 h-4 text-info" />
          {t('simulator.title')}
          {sessions.length > 0 && (
            <span className="bg-accent/20 text-accent text-xs px-1.5 py-0.5 rounded-full">
              {sessions.length}
            </span>
          )}
        </span>
        {collapsed ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
      </button>

      {/* Panel body */}
      {!collapsed && (
        <div className="bg-surface border border-t-0 border-border rounded-b-lg p-4 space-y-4 max-h-[400px] overflow-y-auto">
          {/* Tabs */}
          <div className="flex gap-1 bg-elevated rounded p-0.5">
            {(['ward', 'replay'] as const).map((tabKey) => (
              <button
                key={tabKey}
                onClick={() => setTab(tabKey)}
                className={clsx(
                  'flex-1 text-xs py-1.5 rounded transition-colors capitalize',
                  tab === tabKey ? 'bg-accent/10 text-accent' : 'text-text-muted hover:text-text-secondary',
                )}
              >
                {tabKey === 'ward' ? t('simulator.wardSim') : t('simulator.caseReplay')}
              </button>
            ))}
          </div>

          {/* Ward Tab */}
          {tab === 'ward' && (
            <div className="space-y-3">
              <div>
                <label htmlFor="ward-patients" className="text-[10px] text-text-muted">
                  {t('simulator.patients', { n: wardPatients })}
                </label>
                <input
                  id="ward-patients"
                  type="range"
                  min={4}
                  max={20}
                  value={wardPatients}
                  onChange={(e) => setWardPatients(Number(e.target.value))}
                  className="w-full accent-accent"
                />
              </div>
              <div>
                <label htmlFor="ward-speed" className="text-[10px] text-text-muted">
                  {t('simulator.speed', { n: wardSpeed })}
                </label>
                <input
                  id="ward-speed"
                  type="range"
                  min={1}
                  max={1000}
                  step={10}
                  value={wardSpeed}
                  onChange={(e) => setWardSpeed(Number(e.target.value))}
                  className="w-full accent-accent"
                />
              </div>
              <div>
                <label htmlFor="ward-sepsis" className="text-[10px] text-text-muted">
                  {t('simulator.sepsisPatients', { n: wardSepsis })}
                </label>
                <input
                  id="ward-sepsis"
                  type="range"
                  min={0}
                  max={Math.floor(wardPatients / 2)}
                  value={wardSepsis}
                  onChange={(e) => setWardSepsis(Number(e.target.value))}
                  className="w-full accent-accent"
                />
              </div>
              <button
                onClick={startWard}
                disabled={loading}
                className="w-full flex items-center justify-center gap-1.5 bg-accent/10 text-accent border border-accent/30 rounded py-2 text-xs font-medium hover:bg-accent/20 disabled:opacity-50"
              >
                <Play className="w-3.5 h-3.5" />
                {t('simulator.startWard')}
              </button>
            </div>
          )}

          {/* Replay Tab */}
          {tab === 'replay' && (
            <div className="space-y-3">
              <select
                value={selectedCase ?? ''}
                onChange={(e) => setSelectedCase(e.target.value ? Number(e.target.value) : null)}
                className="w-full bg-elevated border border-border rounded px-3 py-2 text-xs text-text-primary"
              >
                <option value="">{t('simulator.selectCase')}</option>
                {cases.map((c) => (
                  <option key={c.subject_id} value={c.subject_id}>
                    #{c.subject_id} | {c.sex} {c.age_years}y | {c.sepsis_label ? 'SEPSIS' : 'No sepsis'} | {(c.icu_los_hours / 24).toFixed(1)}d
                  </option>
                ))}
              </select>
              <div>
                <label htmlFor="replay-speed" className="text-[10px] text-text-muted">
                  {t('simulator.speed', { n: replaySpeed })}
                </label>
                <input
                  id="replay-speed"
                  type="range"
                  min={1}
                  max={1000}
                  step={10}
                  value={replaySpeed}
                  onChange={(e) => setReplaySpeed(Number(e.target.value))}
                  className="w-full accent-accent"
                />
              </div>
              <button
                onClick={startReplay}
                disabled={loading || !selectedCase}
                className="w-full flex items-center justify-center gap-1.5 bg-accent/10 text-accent border border-accent/30 rounded py-2 text-xs font-medium hover:bg-accent/20 disabled:opacity-50"
              >
                <Play className="w-3.5 h-3.5" />
                {t('simulator.startReplay')}
              </button>
            </div>
          )}

          {/* Active Sessions */}
          {sessions.length > 0 && (
            <div className="space-y-2">
              <h3 className="text-[10px] text-text-muted uppercase tracking-wider">{t('simulator.activeSessions')}</h3>
              {sessions.map((s) => (
                <div
                  key={s.session_id}
                  className="flex items-center justify-between bg-elevated rounded p-2"
                >
                  <div className="text-xs">
                    <span className="text-text-primary capitalize">{s.type}</span>
                    <span className="text-text-muted ml-2">{s.session_id.slice(0, 8)}</span>
                  </div>
                  <button
                    onClick={() => stopSession(s.session_id)}
                    className="p-1 text-text-muted hover:text-danger transition-colors"
                    title={t('simulator.stop')}
                    aria-label={t('simulator.stopSimulation')}
                  >
                    <Square className="w-3.5 h-3.5" />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
