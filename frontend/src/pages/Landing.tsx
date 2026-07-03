import { Link } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { Shield, ArrowRight, Brain, Activity, Globe, Calculator, Plug, Languages } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'

const TEASER_DATA = [
  { label: 'p5', value: 101, fill: '#38b4ff' },
  { label: 'p25', value: 111, fill: '#00ff9d' },
  { label: 'p50', value: 117, fill: '#00ff9d' },
  { label: 'p75', value: 124, fill: '#00ff9d' },
  { label: 'p95', value: 137, fill: '#ffb830' },
]

const FEATURE_ICONS = [Brain, Activity, Globe, Calculator, Plug, Languages]

export function Landing() {
  const { t } = useTranslation()

  const problems = [
    { num: t('landing.problem1Num'), title: t('landing.problem1Title'), desc: t('landing.problem1Desc') },
    { num: t('landing.problem2Num'), title: t('landing.problem2Title'), desc: t('landing.problem2Desc') },
    { num: t('landing.problem3Num'), title: t('landing.problem3Title'), desc: t('landing.problem3Desc') },
  ]

  const features = Array.from({ length: 6 }, (_, i) => ({
    icon: FEATURE_ICONS[i],
    title: t(`landing.feat${i + 1}Title`),
    desc: t(`landing.feat${i + 1}Desc`),
    tag: t(`landing.feat${i + 1}Tag`),
  }))

  const stats = [
    { value: t('landing.stat1Value'), label: t('landing.stat1Label') },
    { value: t('landing.stat2Value'), label: t('landing.stat2Label') },
    { value: t('landing.stat3Value'), label: t('landing.stat3Label') },
  ]

  return (
    <div className="min-h-screen bg-void text-text-primary font-mono">
      {/* Nav */}
      <nav className="fixed top-0 inset-x-0 z-50 bg-void/90 backdrop-blur-md border-b border-border h-14 flex items-center justify-between px-6">
        <Link to="/" className="flex items-center gap-2 font-heading font-bold text-sm tracking-wide">
          <Shield className="w-5 h-5 text-accent" />
          <span className="text-accent">{t('app.title')}</span>
        </Link>
        <div className="flex items-center gap-6">
          <Link to="/pricing" className="text-xs text-text-secondary hover:text-accent transition-colors hidden sm:inline">
            {t('landing.viewPricing')}
          </Link>
          <Link to="/login" className="text-xs font-bold bg-accent text-void px-4 py-2 rounded hover:bg-accent-dim transition-colors">
            {t('landing.signIn')}
          </Link>
        </div>
      </nav>

      {/* Hero */}
      <section className="relative min-h-screen flex items-center justify-center pt-14 px-6 overflow-hidden">
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-[radial-gradient(circle,rgba(0,255,157,0.06)_0%,transparent_70%)] pointer-events-none" />
        <div className="relative z-10 max-w-3xl text-center">
          <div className="inline-flex items-center gap-2 bg-accent/10 border border-accent/20 rounded-full px-4 py-1.5 text-xs text-accent mb-6">
            <span className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse" />
            AI-Powered Clinical Decision Support
          </div>
          <h1 className="font-heading text-[clamp(2rem,6vw,3.75rem)] font-extrabold leading-[1.08] mb-5 tracking-tight">
            {t('landing.heroTitle1')}{' '}
            <em className="not-italic text-accent" style={{ textShadow: '0 0 40px rgba(0,255,157,0.3)' }}>
              {t('landing.heroTitle2')}
            </em>
          </h1>
          <p className="text-sm sm:text-base text-text-secondary max-w-xl mx-auto mb-8 leading-relaxed">
            {t('landing.heroSub')}
          </p>
          <div className="flex gap-3 justify-center flex-wrap">
            <Link to="/login" className="inline-flex items-center gap-2 bg-accent text-void font-bold text-sm px-8 py-3.5 rounded hover:bg-accent-dim transition-all hover:shadow-[0_0_30px_rgba(0,255,157,0.3)]">
              {t('landing.getStarted')} <ArrowRight className="w-4 h-4" />
            </Link>
            <Link to="/pricing" className="inline-flex items-center gap-2 border border-border text-text-primary text-sm px-8 py-3.5 rounded hover:border-accent hover:text-accent transition-colors">
              {t('landing.viewPricing')}
            </Link>
          </div>
          <div className="flex gap-10 justify-center mt-12 flex-wrap">
            {stats.map((s) => (
              <div key={s.label} className="text-center">
                <div className="font-heading text-2xl sm:text-3xl font-extrabold text-accent">{s.value}</div>
                <div className="text-[10px] text-text-muted uppercase tracking-widest mt-1">{s.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Problem */}
      <section className="px-6 py-20 bg-surface">
        <div className="max-w-4xl mx-auto">
          <p className="text-[10px] font-bold tracking-[0.18em] uppercase text-accent mb-3">{t('landing.problemLabel')}</p>
          <h2 className="font-heading text-2xl sm:text-3xl font-bold mb-10">{t('landing.problemTitle')}</h2>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            {problems.map((p) => (
              <div key={p.num} className="bg-elevated border border-border rounded-lg p-6 border-t-2 border-t-danger">
                <div className="font-heading text-3xl font-extrabold text-danger mb-2">{p.num}</div>
                <h3 className="font-heading text-sm font-semibold mb-2">{p.title}</h3>
                <p className="text-xs text-text-secondary leading-relaxed">{p.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="px-6 py-20">
        <div className="max-w-4xl mx-auto">
          <p className="text-[10px] font-bold tracking-[0.18em] uppercase text-accent mb-3">{t('landing.featuresLabel')}</p>
          <h2 className="font-heading text-2xl sm:text-3xl font-bold mb-10">{t('landing.featuresTitle')}</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {features.map((f) => {
              const Icon = f.icon
              return (
                <div key={f.title} className="bg-surface border border-border rounded-lg p-6 hover:border-accent hover:-translate-y-0.5 transition-all">
                  <Icon className="w-7 h-7 text-accent mb-4" />
                  <h3 className="font-heading text-sm font-semibold mb-2">{f.title}</h3>
                  <p className="text-xs text-text-secondary leading-relaxed mb-3">{f.desc}</p>
                  <span className="inline-block text-[9px] font-bold tracking-widest uppercase bg-accent/10 text-accent border border-accent/20 rounded px-2 py-0.5">
                    {f.tag}
                  </span>
                </div>
              )
            })}
          </div>
        </div>
      </section>

      {/* NHANES Teaser */}
      <section className="px-6 py-20 bg-surface">
        <div className="max-w-4xl mx-auto">
          <p className="text-[10px] font-bold tracking-[0.18em] uppercase text-accent mb-3">{t('landing.nhanesLabel')}</p>
          <h2 className="font-heading text-2xl sm:text-3xl font-bold mb-4">{t('landing.nhanesTitle')}</h2>
          <p className="text-sm text-text-secondary max-w-xl mb-8 leading-relaxed">{t('landing.nhanesSub')}</p>
          <div className="bg-elevated border border-border rounded-lg p-6 max-w-md">
            <p className="text-xs text-text-muted mb-3">Systolic BP Percentiles — Male 40-49</p>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={TEASER_DATA}>
                <XAxis dataKey="label" tick={{ fill: '#4a6080', fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: '#4a6080', fontSize: 11 }} axisLine={false} tickLine={false} domain={[80, 160]} />
                <Tooltip contentStyle={{ background: '#111d2e', border: '1px solid rgba(255,255,255,0.06)', borderRadius: 6, fontSize: 12 }} />
                <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                  {TEASER_DATA.map((d, i) => (
                    <Cell key={i} fill={d.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <Link to="/login" className="inline-flex items-center gap-2 text-accent text-sm mt-6 hover:underline">
            {t('landing.nhanesExplore')} <ArrowRight className="w-4 h-4" />
          </Link>
        </div>
      </section>

      {/* Compliance */}
      <section className="px-6 py-16">
        <div className="max-w-4xl mx-auto flex flex-wrap gap-4 justify-center">
          {[t('landing.hipaa'), t('landing.soc2'), t('landing.fhir')].map((badge) => (
            <div key={badge} className="flex items-center gap-2 bg-surface border border-border rounded px-4 py-2 text-xs text-text-secondary">
              <Shield className="w-4 h-4 text-accent" />
              <strong className="text-text-primary">{badge}</strong>
            </div>
          ))}
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border px-6 py-8">
        <div className="max-w-4xl mx-auto flex items-center justify-between text-xs text-text-muted">
          <span>&copy; {new Date().getFullYear()} {t('landing.copyright')}</span>
          <div className="flex gap-6">
            <Link to="/pricing" className="hover:text-accent transition-colors">{t('landing.viewPricing')}</Link>
            <Link to="/login" className="hover:text-accent transition-colors">{t('landing.signIn')}</Link>
          </div>
        </div>
      </footer>
    </div>
  )
}
