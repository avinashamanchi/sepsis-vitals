import { ArrowRight, Building2, Check, ClipboardCheck, Database, Users } from 'lucide-react'
import { PublicShell } from '../components/PublicShell'

const requirements = [
  {
    icon: Database,
    title: 'Usable historical data',
    body: 'Adult inpatient encounters with timestamped vitals, relevant labs, outcomes, and enough provenance to build a defensible reference standard.',
  },
  {
    icon: Users,
    title: 'Clinical ownership',
    body: 'A clinical safety lead, data lead, and operational sponsor who can define the population and review failure modes.',
  },
  {
    icon: ClipboardCheck,
    title: 'Governance before transfer',
    body: 'IRB or quality-improvement determination, data-use agreement, security review, and a named owner for every approval.',
  },
  {
    icon: Building2,
    title: 'A real workflow question',
    body: 'A specific ward, review cadence, and escalation workflow—not a generic request to “add AI” to the hospital.',
  },
]

const deliverables = [
  'Data-quality and cohort-readiness report',
  'Locked statistical analysis plan',
  'Site-specific performance and calibration report',
  'Subgroup and missingness analysis',
  'Alert-burden and workflow simulation',
  'Go / revise / stop recommendation',
]

export function Pilot() {
  return (
    <PublicShell>
      <section className="px-5 pb-20 pt-16 sm:px-8 sm:pt-24">
        <div className="mx-auto max-w-5xl">
          <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-accent">Founding validation partners</p>
          <h1 className="mt-4 max-w-4xl font-heading text-[clamp(2.6rem,6vw,4.75rem)] font-extrabold leading-[1.01] tracking-[-0.05em]">
            Before you buy the software,{' '}
            <span className="text-accent">test the premise.</span>
          </h1>
          <p className="mt-7 max-w-2xl text-base leading-7 text-text-secondary">
            The pilot is a structured evaluation engagement for hospitals and research
            groups. It is not a free trial, a live alert deployment, or a claim that the
            model already works in your population.
          </p>
          <a
            href="mailto:sales@sepsisvitals.com?subject=Founding%20validation%20partner"
            className="mt-9 inline-flex items-center gap-2 rounded-md bg-accent px-5 py-3 text-sm font-bold text-void"
          >
            Request a fit review
            <ArrowRight className="h-4 w-4" aria-hidden="true" />
          </a>
        </div>
      </section>

      <section className="border-y border-border bg-surface/55 px-5 py-24 sm:px-8">
        <div className="mx-auto max-w-5xl">
          <div className="grid gap-10 lg:grid-cols-[.75fr_1.25fr]">
            <div>
              <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-accent">Good fit</p>
              <h2 className="mt-3 font-heading text-3xl font-bold">What a partner needs</h2>
              <p className="mt-4 text-sm leading-6 text-text-secondary">
                If these pieces are not available, the honest answer is to fix the study
                foundation before evaluating a model.
              </p>
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              {requirements.map(({ icon: Icon, title, body }) => (
                <article key={title} className="rounded-xl border border-border bg-void p-6">
                  <Icon className="h-5 w-5 text-accent" aria-hidden="true" />
                  <h3 className="mt-4 font-heading text-base font-semibold">{title}</h3>
                  <p className="mt-2 text-xs leading-6 text-text-secondary">{body}</p>
                </article>
              ))}
            </div>
          </div>
        </div>
      </section>

      <section className="px-5 py-24 sm:px-8">
        <div className="mx-auto grid max-w-5xl gap-10 lg:grid-cols-2">
          <div>
            <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-accent">What you receive</p>
            <h2 className="mt-3 font-heading text-3xl font-bold">A decision package, not a vanity metric.</h2>
            <p className="mt-5 text-sm leading-7 text-text-secondary">
              Every engagement ends with enough evidence to choose the next step. A negative
              or inconclusive result is a valid outcome and will be reported plainly.
            </p>
          </div>
          <div className="rounded-xl border border-border-bright bg-surface p-7">
            <ul className="space-y-4">
              {deliverables.map((item) => (
                <li key={item} className="flex items-center gap-3 text-sm text-text-secondary">
                  <span className="grid h-6 w-6 shrink-0 place-items-center rounded-full bg-accent/10">
                    <Check className="h-3.5 w-3.5 text-accent" aria-hidden="true" />
                  </span>
                  {item}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </section>

      <section className="px-5 pb-24 sm:px-8">
        <div className="mx-auto max-w-5xl rounded-2xl border border-accent/25 bg-accent/7 p-8 sm:p-10">
          <h2 className="font-heading text-2xl font-bold">What we need in the first email</h2>
          <p className="mt-3 max-w-3xl text-sm leading-6 text-text-secondary">
            Country and health system, target ward, approximate eligible encounter count,
            available vitals/labs/outcomes, desired study timeline, and the names or roles
            of the clinical and data leads.
          </p>
          <a
            href="mailto:sales@sepsisvitals.com?subject=Sepsis%20Vitals%20pilot%20fit%20review"
            className="mt-6 inline-flex items-center gap-2 text-sm font-bold text-accent hover:underline"
          >
            sales@sepsisvitals.com
            <ArrowRight className="h-4 w-4" aria-hidden="true" />
          </a>
        </div>
      </section>
    </PublicShell>
  )
}
