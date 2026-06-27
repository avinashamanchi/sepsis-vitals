import { useState, useCallback } from 'react'
import { Shield, AlertTriangle, Scale, FileCheck } from 'lucide-react'
import { useTranslation } from 'react-i18next'

const EULA_VERSION = '1.0.0'
const STORAGE_KEY = 'sv_eula_accepted'

function hasAcceptedEula(): boolean {
  try {
    const val = localStorage.getItem(STORAGE_KEY)
    if (!val) return false
    const parsed = JSON.parse(val)
    return parsed.version === EULA_VERSION
  } catch {
    return false
  }
}

function acceptEula(): void {
  localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify({ version: EULA_VERSION, acceptedAt: new Date().toISOString() }),
  )
}

export function EulaGate({ children }: { children: React.ReactNode }) {
  const { t } = useTranslation()
  const [accepted, setAccepted] = useState(hasAcceptedEula)
  const [scrolledToBottom, setScrolledToBottom] = useState(false)
  const [checked, setChecked] = useState(false)

  const handleScroll = useCallback((e: React.UIEvent<HTMLDivElement>) => {
    const el = e.currentTarget
    if (el.scrollHeight - el.scrollTop - el.clientHeight < 40) {
      setScrolledToBottom(true)
    }
  }, [])

  const handleAccept = () => {
    acceptEula()
    setAccepted(true)
  }

  if (accepted) return <>{children}</>

  return (
    <div className="fixed inset-0 z-[100] bg-void flex items-center justify-center p-4">
      <div className="w-full max-w-2xl animate-fade-in">
        {/* Header */}
        <div className="text-center mb-6">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-accent/10 border border-accent/20 mb-4">
            <Shield className="w-8 h-8 text-accent" />
          </div>
          <h1 className="font-heading text-2xl font-bold text-text-primary">
            {t('eula.title')}
          </h1>
          <p className="text-sm text-text-secondary mt-1">
            {t('eula.subtitle')}
          </p>
        </div>

        {/* EULA Content */}
        <div
          onScroll={handleScroll}
          className="bg-surface border border-border rounded-xl max-h-[55vh] overflow-y-auto p-6 space-y-5 text-sm leading-relaxed text-text-secondary"
        >
          <div className="flex items-start gap-3 p-4 bg-danger/8 border border-danger/20 rounded-lg">
            <AlertTriangle className="w-5 h-5 text-danger flex-shrink-0 mt-0.5" />
            <div>
              <p className="font-semibold text-danger text-sm">{t('eula.criticalNotice')}</p>
              <p className="text-text-secondary mt-1">{t('eula.criticalBody')}</p>
            </div>
          </div>

          <section>
            <h3 className="text-text-primary font-semibold flex items-center gap-2 mb-2">
              <FileCheck className="w-4 h-4 text-accent" />
              {t('eula.section1Title')}
            </h3>
            <p>{t('eula.section1Body')}</p>
          </section>

          <section>
            <h3 className="text-text-primary font-semibold flex items-center gap-2 mb-2">
              <Scale className="w-4 h-4 text-accent" />
              {t('eula.section2Title')}
            </h3>
            <p>{t('eula.section2Body')}</p>
            <p className="mt-2">
              <strong className="text-text-primary">{t('eula.section2Bold')}</strong>
            </p>
          </section>

          <section>
            <h3 className="text-text-primary font-semibold mb-2">{t('eula.section3Title')}</h3>
            <p>
              {t('eula.section3Body')}{' '}
              <strong className="text-text-primary">{t('eula.section3Ruo')}</strong>
              {t('eula.section3Rest')}
            </p>
          </section>

          <section>
            <h3 className="text-text-primary font-semibold mb-2">{t('eula.section4Title')}</h3>
            <p>{t('eula.section4Body')}</p>
          </section>

          <section>
            <h3 className="text-text-primary font-semibold mb-2">{t('eula.section5Title')}</h3>
            <p>{t('eula.section5Body')}</p>
          </section>

          <section>
            <h3 className="text-text-primary font-semibold mb-2">{t('eula.section6Title')}</h3>
            <p>{t('eula.section6Body')}</p>
          </section>

          <section>
            <h3 className="text-text-primary font-semibold mb-2">{t('eula.section7Title')}</h3>
            <p>{t('eula.section7Body')}</p>
          </section>

          <p className="text-text-muted text-xs pt-2 border-t border-border">
            {t('eula.version')}
          </p>
        </div>

        {/* Acceptance Controls */}
        <div className="mt-5 space-y-4">
          {!scrolledToBottom && (
            <p className="text-xs text-text-muted text-center">
              {t('eula.scrollHint')}
            </p>
          )}

          <label
            className={`flex items-start gap-3 cursor-pointer select-none transition-opacity ${
              scrolledToBottom ? 'opacity-100' : 'opacity-40 pointer-events-none'
            }`}
          >
            <input
              type="checkbox"
              checked={checked}
              onChange={(e) => setChecked(e.target.checked)}
              disabled={!scrolledToBottom}
              className="mt-1 w-4 h-4 accent-accent rounded border-border bg-surface"
            />
            <span className="text-sm text-text-secondary">
              {t('eula.checkboxLabel')}
            </span>
          </label>

          <div className="flex gap-3">
            <button
              onClick={handleAccept}
              disabled={!checked}
              className={`flex-1 py-3 rounded-lg font-heading font-semibold text-sm transition-all ${
                checked
                  ? 'bg-accent text-void hover:bg-accent-dim cursor-pointer glow-accent'
                  : 'bg-elevated text-text-muted cursor-not-allowed'
              }`}
            >
              {t('eula.agreeButton')}
            </button>
          </div>

          <p className="text-[11px] text-text-muted text-center">
            {t('eula.footerNote')}
          </p>
        </div>
      </div>
    </div>
  )
}
