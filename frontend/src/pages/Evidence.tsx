import { AlertTriangle, ArrowRight, Check, CircleDashed, FlaskConical, X } from 'lucide-react'
import { Link } from 'react-router-dom'
import { PublicShell } from '../components/PublicShell'

const facts = [
  ['Intended use today', 'Software research, retrospective analysis, and prospective silent-mode evaluation'],
  ['Training source', 'Procedurally generated adult trajectories using assumptions encoded in source'],
  ['External validation', 'Not completed'],
  ['Regulatory status', 'Investigational; not FDA-cleared or CE-marked'],
  ['Clinical use', 'Not authorized for diagnosis, treatment, or autonomous alerting'],
  ['Current model version', 'Gradient boosting development baseline v2.0'],
]

const gates = [
  {
    state: 'done',
    title: 'Engineering baseline',
    body: 'Reproducible model artifact, versioned metadata, audit trail, clinical-score tests, and data-ingestion adapters.',
  },
  {
    state: 'current',
    title: 'Retrospective external validation',
    body: 'Evaluate discrimination, calibration, subgroup performance, missingness, transportability, and alert burden on unseen hospital data.',
  },
  {
    state: 'next',
    title: 'Prospective silent-mode study',
    body: 'Measure lead time, workflow fit, false-alert burden, and human-factors risks without influencing patient care.',
  },
  {
    state: 'next',
    title: 'Regulatory and clinical deployment decision',
    body: 'Define the intended use and risk pathway with qualified regulatory counsel before any live clinical positioning.',
  },
]

const noClaims = [
  'We do not claim six-hour earlier detection.',
  'We do not claim mortality reduction or lives saved.',
  'We do not describe synthetic-test performance as clinical performance.',
  'We do not call the product HIPAA compliant or SOC 2 ready without independent evidence.',
]

export function Evidence() {
  return (
    <PublicShell>
      <section className="px-5 pb-16 pt-16 sm:px-8 sm:pb-20 sm:pt-24">
        <div className="mx-auto max-w-4xl">
          <div className="inline-flex items-center gap-2 rounded-full border border-info/20 bg-info/8 px-3 py-1.5 text-[11px] font-semibold text-info">
            <FlaskConical className="h-3.5 w-3.5" aria-hidden="true" />
            Evidence ledger · updated July 2026
          </div>
          <h1 className="mt-6 max-w-3xl font-heading text-[clamp(2.5rem,6vw,4.5rem)] font-extrabold leading-[1.02] tracking-[-0.045em]">
            Trust starts with saying what the model{' '}
            <span className="text-accent">cannot prove.</span>
          </h1>
          <p className="mt-6 max-w-2xl text-base leading-7 text-text-secondary">
            This page is the product’s source of truth. Marketing language must never outrun
            the evidence recorded here.
          </p>
        </div>
      </section>

      <section className="px-5 pb-24 sm:px-8">
        <div className="mx-auto grid max-w-4xl overflow-hidden rounded-xl border border-border lg:grid-cols-2">
          {facts.map(([label, value]) => (
            <div key={label} className="border-b border-border p-5 last:border-b-0 lg:border-r lg:odd:border-r lg:even:border-r-0">
              <p className="text-[10px] font-bold uppercase tracking-[0.14em] text-text-muted">{label}</p>
              <p className="mt-2 text-sm leading-6 text-text-primary">{value}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="border-y border-border bg-surface/55 px-5 py-24 sm:px-8">
        <div className="mx-auto max-w-4xl">
          <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-accent">Validation roadmap</p>
          <h2 className="mt-3 font-heading text-3xl font-bold">Four gates. No shortcuts.</h2>

          <div className="mt-10">
            {gates.map((gate, index) => (
              <div key={gate.title} className="relative grid grid-cols-[2.5rem_1fr] gap-4 pb-9 last:pb-0">
                {index < gates.length - 1 && <div className="absolute left-[1.18rem] top-8 h-[calc(100%-1.25rem)] w-px bg-border" />}
                <div className={`relative z-10 grid h-9 w-9 place-items-center rounded-full border ${
                  gate.state === 'done'
                    ? 'border-accent/30 bg-accent/10 text-accent'
                    : gate.state === 'current'
                      ? 'border-warning/35 bg-warning/10 text-warning'
                      : 'border-border-bright bg-void text-text-muted'
                }`}>
                  {gate.state === 'done' ? <Check className="h-4 w-4" /> : <CircleDashed className="h-4 w-4" />}
                </div>
                <div className="pt-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <h3 className="font-heading text-base font-semibold">{gate.title}</h3>
                    <span className={`rounded-full px-2 py-0.5 text-[9px] font-bold uppercase tracking-wider ${
                      gate.state === 'done'
                        ? 'bg-accent/10 text-accent'
                        : gate.state === 'current'
                          ? 'bg-warning/10 text-warning'
                          : 'bg-elevated text-text-muted'
                    }`}>
                      {gate.state === 'done' ? 'complete' : gate.state === 'current' ? 'current gate' : 'not started'}
                    </span>
                  </div>
                  <p className="mt-2 text-sm leading-6 text-text-secondary">{gate.body}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="px-5 py-24 sm:px-8">
        <div className="mx-auto grid max-w-4xl gap-8 lg:grid-cols-2">
          <div className="rounded-xl border border-danger/20 bg-danger/5 p-7">
            <div className="flex items-center gap-2 text-danger">
              <AlertTriangle className="h-5 w-5" aria-hidden="true" />
              <h2 className="font-heading text-lg font-semibold">Claims we will not make</h2>
            </div>
            <ul className="mt-5 space-y-3">
              {noClaims.map((claim) => (
                <li key={claim} className="flex gap-2.5 text-sm leading-6 text-text-secondary">
                  <X className="mt-1 h-3.5 w-3.5 shrink-0 text-danger" aria-hidden="true" />
                  {claim}
                </li>
              ))}
            </ul>
          </div>

          <div className="rounded-xl border border-accent/20 bg-accent/5 p-7">
            <h2 className="font-heading text-lg font-semibold">What a credible result looks like</h2>
            <p className="mt-4 text-sm leading-6 text-text-secondary">
              A successful study is not simply a high AUROC. It shows calibration at the
              intended operating point, acceptable performance across subgroups and sites,
              manageable alerts per patient-day, robust behavior with missing data, and a
              workflow clinicians can understand.
            </p>
            <Link to="/pilot" className="mt-6 inline-flex items-center gap-2 text-sm font-semibold text-accent hover:underline">
              Review the pilot design
              <ArrowRight className="h-4 w-4" aria-hidden="true" />
            </Link>
          </div>
        </div>
      </section>
    </PublicShell>
  )
}
