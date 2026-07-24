import {
  ArrowRight,
  Check,
  ChevronRight,
  CircleAlert,
  Database,
  FileCheck2,
  Gauge,
  Languages,
  Network,
  ShieldCheck,
  Stethoscope,
} from 'lucide-react'
import { Link } from 'react-router-dom'
import { PublicShell } from '../components/PublicShell'

const wardPatients = [
  { bed: 'D-12', id: 'SV-1042', risk: 'Needs review', score: '0.68', tone: 'danger', trend: '+14%' },
  { bed: 'D-08', id: 'SV-1031', risk: 'Watch', score: '0.41', tone: 'warning', trend: '+4%' },
  { bed: 'D-03', id: 'SV-1018', risk: 'Stable', score: '0.12', tone: 'accent', trend: '−2%' },
]

const toneText = {
  danger: 'text-danger',
  warning: 'text-warning',
  accent: 'text-accent',
} as const

const capabilities = [
  {
    icon: Gauge,
    title: 'One prioritized ward view',
    body: 'Bring routine vitals, labs, score changes, and data freshness into one review queue built for shift work.',
  },
  {
    icon: FileCheck2,
    title: 'Evidence before automation',
    body: 'Measure discrimination, calibration, subgroup performance, lead time, and alert burden before any live workflow.',
  },
  {
    icon: Network,
    title: 'Fits existing systems',
    body: 'FHIR R4 and HL7v2 ingestion, auditable prediction records, and degraded-network support for constrained sites.',
  },
  {
    icon: Languages,
    title: 'Designed across settings',
    body: 'Six-language interface support with site-specific thresholds and workflows planned around local validation.',
  },
]

const pilotSteps = [
  {
    number: '01',
    title: 'Connect a retrospective dataset',
    body: 'Map available observations, check missingness and label quality, and agree on the intended population.',
  },
  {
    number: '02',
    title: 'Run a locked evaluation',
    body: 'Test the model on unseen, site-specific data with pre-agreed performance and safety criteria.',
  },
  {
    number: '03',
    title: 'Enter silent mode—or stop',
    body: 'Only qualified sites move to prospective observation. No alerts influence care until the evidence supports it.',
  },
]

export function Landing() {
  return (
    <PublicShell>
      <section className="relative overflow-hidden px-5 pb-24 pt-16 sm:px-8 sm:pb-28 sm:pt-24">
        <div className="landing-grid absolute inset-0 opacity-40" aria-hidden="true" />
        <div className="landing-glow absolute left-[55%] top-0 h-[520px] w-[520px] -translate-y-1/3 rounded-full" aria-hidden="true" />

        <div className="relative mx-auto grid max-w-6xl items-center gap-14 lg:grid-cols-[1.05fr_.95fr]">
          <div>
            <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-warning/25 bg-warning/8 px-3 py-1.5 text-[11px] font-semibold text-warning">
              <CircleAlert className="h-3.5 w-3.5" aria-hidden="true" />
              Investigational · research and pilot use only
            </div>
            <h1 className="max-w-3xl font-heading text-[clamp(2.65rem,6vw,5rem)] font-extrabold leading-[0.98] tracking-[-0.055em]">
              Find the patients who need a{' '}
              <span className="text-accent">closer look.</span>
            </h1>
            <p className="mt-7 max-w-2xl text-base leading-7 text-text-secondary sm:text-lg">
              Sepsis Vitals turns fragmented observations into a prioritized review queue,
              then helps hospitals prove whether the signal is useful on their own data
              before it reaches a clinical workflow.
            </p>
            <div className="mt-9 flex flex-col gap-3 sm:flex-row">
              <Link
                to="/pilot"
                className="inline-flex items-center justify-center gap-2 rounded-md bg-accent px-5 py-3 text-sm font-bold text-void transition-transform hover:-translate-y-0.5"
              >
                Evaluate a pilot
                <ArrowRight className="h-4 w-4" aria-hidden="true" />
              </Link>
              <Link
                to="/evidence"
                className="inline-flex items-center justify-center gap-2 rounded-md border border-border-bright bg-surface/70 px-5 py-3 text-sm font-semibold text-text-primary hover:border-text-muted"
              >
                See what is—and is not—validated
              </Link>
            </div>
            <div className="mt-8 flex flex-wrap gap-x-6 gap-y-3 text-[11px] text-text-muted">
              {['Synthetic baseline disclosed', 'Local validation required', 'No autonomous decisions'].map((item) => (
                <span key={item} className="flex items-center gap-1.5">
                  <Check className="h-3.5 w-3.5 text-accent" aria-hidden="true" />
                  {item}
                </span>
              ))}
            </div>
          </div>

          <div className="relative">
            <div className="absolute -inset-6 rounded-[2rem] bg-accent/5 blur-3xl" aria-hidden="true" />
            <div className="relative overflow-hidden rounded-2xl border border-border-bright bg-[#07101c] shadow-2xl">
              <div className="flex items-center justify-between border-b border-border px-5 py-4">
                <div>
                  <p className="font-heading text-sm font-semibold">Ward review queue</p>
                  <p className="mt-1 text-[10px] text-text-muted">Synthetic demonstration</p>
                </div>
                <span className="flex items-center gap-1.5 rounded-full border border-accent/20 bg-accent/8 px-2.5 py-1 text-[10px] text-accent">
                  <span className="h-1.5 w-1.5 rounded-full bg-accent" />
                  data current
                </span>
              </div>
              <div className="grid grid-cols-3 border-b border-border bg-surface/60">
                {[
                  ['18', 'patients'],
                  ['2', 'review'],
                  ['94%', 'data complete'],
                ].map(([value, label]) => (
                  <div key={label} className="border-r border-border px-4 py-4 last:border-r-0">
                    <p className="font-heading text-xl font-bold text-text-primary">{value}</p>
                    <p className="mt-1 text-[9px] uppercase tracking-[0.14em] text-text-muted">{label}</p>
                  </div>
                ))}
              </div>
              <div className="space-y-2 p-3">
                {wardPatients.map((patient) => (
                  <div
                    key={patient.id}
                    className="grid grid-cols-[auto_1fr_auto] items-center gap-3 rounded-lg border border-border bg-surface/70 p-3.5"
                  >
                    <div className={`risk-dot risk-dot-${patient.tone}`} aria-hidden="true" />
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="font-heading text-xs font-semibold">{patient.bed}</span>
                        <span className="text-[10px] text-text-muted">{patient.id}</span>
                      </div>
                      <p className={`mt-1 text-[10px] ${toneText[patient.tone as keyof typeof toneText]}`}>
                        {patient.risk}
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="font-heading text-lg font-bold tabular-nums">{patient.score}</p>
                      <p className="text-[10px] text-text-muted">{patient.trend} / 2h</p>
                    </div>
                  </div>
                ))}
              </div>
              <div className="flex items-start gap-2.5 border-t border-warning/15 bg-warning/5 px-5 py-3 text-[10px] leading-relaxed text-warning/90">
                <CircleAlert className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                Model output shown for evaluation only. Clinicians follow existing standards of care.
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="border-y border-border bg-surface/45 px-5 py-7 sm:px-8">
        <div className="mx-auto grid max-w-6xl gap-6 text-xs text-text-muted sm:grid-cols-3">
          <div className="flex items-center gap-3">
            <Database className="h-5 w-5 text-info" aria-hidden="true" />
            <span><strong className="text-text-primary">FHIR R4 + HL7v2</strong><br />ingestion paths</span>
          </div>
          <div className="flex items-center gap-3">
            <ShieldCheck className="h-5 w-5 text-info" aria-hidden="true" />
            <span><strong className="text-text-primary">Traceable outputs</strong><br />with model versioning</span>
          </div>
          <div className="flex items-center gap-3">
            <Languages className="h-5 w-5 text-info" aria-hidden="true" />
            <span><strong className="text-text-primary">Six interface languages</strong><br />for multi-site research</span>
          </div>
        </div>
      </section>

      <section className="px-5 py-24 sm:px-8">
        <div className="mx-auto max-w-6xl">
          <div className="max-w-2xl">
            <p className="mb-3 text-[10px] font-bold uppercase tracking-[0.2em] text-accent">The product</p>
            <h2 className="font-heading text-3xl font-bold tracking-tight sm:text-4xl">
              Not another score. A safer way to evaluate one.
            </h2>
            <p className="mt-5 text-sm leading-7 text-text-secondary">
              Early-warning products fail when a promising AUROC gets mistaken for a usable
              clinical system. Sepsis Vitals is built around the harder work: data quality,
              workflow fit, alert burden, transparency, and local evidence.
            </p>
          </div>

          <div className="mt-12 grid gap-px overflow-hidden rounded-xl border border-border bg-border sm:grid-cols-2">
            {capabilities.map(({ icon: Icon, title, body }) => (
              <article key={title} className="bg-void p-7 sm:p-8">
                <Icon className="h-6 w-6 text-accent" aria-hidden="true" />
                <h3 className="mt-5 font-heading text-lg font-semibold">{title}</h3>
                <p className="mt-3 text-sm leading-6 text-text-secondary">{body}</p>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="border-y border-border bg-surface/55 px-5 py-24 sm:px-8">
        <div className="mx-auto max-w-6xl">
          <div className="grid gap-12 lg:grid-cols-[.8fr_1.2fr]">
            <div>
              <p className="mb-3 text-[10px] font-bold uppercase tracking-[0.2em] text-accent">Pilot path</p>
              <h2 className="font-heading text-3xl font-bold tracking-tight">
                Earn the right to go live.
              </h2>
              <p className="mt-5 text-sm leading-7 text-text-secondary">
                The correct first sale is a paid validation partnership—not a per-bed
                subscription attached to unproven outcome claims.
              </p>
              <Link to="/pilot" className="mt-7 inline-flex items-center gap-2 text-sm font-semibold text-accent hover:underline">
                Review pilot requirements
                <ChevronRight className="h-4 w-4" aria-hidden="true" />
              </Link>
            </div>
            <div className="divide-y divide-border border-y border-border">
              {pilotSteps.map((step) => (
                <div key={step.number} className="grid gap-4 py-6 sm:grid-cols-[3rem_1fr]">
                  <span className="font-heading text-sm font-bold text-accent">{step.number}</span>
                  <div>
                    <h3 className="font-heading text-base font-semibold">{step.title}</h3>
                    <p className="mt-2 text-sm leading-6 text-text-secondary">{step.body}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      <section className="px-5 py-24 sm:px-8">
        <div className="mx-auto grid max-w-6xl gap-8 rounded-2xl border border-border-bright bg-[linear-gradient(135deg,rgba(0,255,157,0.09),rgba(56,180,255,0.035)_55%,transparent)] p-8 sm:p-12 lg:grid-cols-[1fr_auto] lg:items-center">
          <div>
            <div className="mb-5 flex h-11 w-11 items-center justify-center rounded-xl border border-accent/25 bg-accent/10">
              <Stethoscope className="h-5 w-5 text-accent" aria-hidden="true" />
            </div>
            <h2 className="font-heading text-2xl font-bold sm:text-3xl">Have the data to test this properly?</h2>
            <p className="mt-3 max-w-2xl text-sm leading-6 text-text-secondary">
              We are looking for clinical and research partners with retrospective adult
              inpatient data, a clear governance path, and the willingness to publish negative
              as well as positive results.
            </p>
          </div>
          <a
            href="mailto:sales@sepsisvitals.com?subject=Sepsis%20Vitals%20pilot%20evaluation"
            className="inline-flex items-center justify-center gap-2 rounded-md bg-accent px-5 py-3 text-sm font-bold text-void"
          >
            Start a pilot conversation
            <ArrowRight className="h-4 w-4" aria-hidden="true" />
          </a>
        </div>
      </section>
    </PublicShell>
  )
}
