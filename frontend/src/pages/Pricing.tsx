import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { Shield, Check, X, ArrowRight } from 'lucide-react'

interface Tier {
  key: string
  monthly: number
  annual: number
  featured?: boolean
}

const TIERS: Tier[] = [
  { key: 'starter', monthly: 15, annual: 150 },
  { key: 'professional', monthly: 32, annual: 320, featured: true },
  { key: 'enterprise', monthly: 50, annual: 500 },
]

const FEATURE_MATRIX_KEYS = [
  'qsofa',
  'news2',
  'mlPredictions',
  'customModels',
  'emailAlerts',
  'realtimeWs',
  'escalationChains',
  'basicDashboard',
  'nhanes',
  'fhirIntegration',
  'communitySupport',
  'prioritySupport',
  'dedicatedSla',
] as const

// true = included, false = not included
const FEATURE_AVAILABILITY: Record<string, [boolean, boolean, boolean]> = {
  qsofa:            [true,  true,  true],
  news2:            [true,  true,  true],
  mlPredictions:    [false, true,  true],
  customModels:     [false, false, true],
  emailAlerts:      [true,  true,  true],
  realtimeWs:       [false, true,  true],
  escalationChains: [false, false, true],
  basicDashboard:   [true,  true,  true],
  nhanes:           [false, true,  true],
  fhirIntegration:  [false, false, true],
  communitySupport: [true,  true,  true],
  prioritySupport:  [false, true,  true],
  dedicatedSla:     [false, false, true],
}

const COST_PER_SEPSIS = 35_000
const SAVINGS_RATE = 0.25
const MORTALITY_REDUCTION = 0.20

export function Pricing() {
  const { t } = useTranslation()
  const [annual, setAnnual] = useState(true)
  const [beds, setBeds] = useState(200)
  const [sepsisCases, setSepsisCases] = useState(120)

  const savings = sepsisCases * COST_PER_SEPSIS * SAVINGS_RATE
  const livesSaved = Math.round(sepsisCases * MORTALITY_REDUCTION)
  const cost = beds * TIERS[1].monthly * 12
  const roi = cost > 0 ? Math.round(((savings - cost) / cost) * 100) : 0

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

      {/* Header */}
      <section className="pt-28 pb-12 px-6 text-center">
        <h1 className="font-heading text-[clamp(2rem,5vw,3rem)] font-extrabold tracking-tight mb-4">
          {t('pricing.headline')}
        </h1>
        <p className="text-sm sm:text-base text-text-secondary max-w-lg mx-auto">
          {t('pricing.subtext')}
        </p>
      </section>

      {/* Toggle */}
      <div className="flex items-center justify-center gap-3 mb-12">
        <span className={`text-xs ${!annual ? 'text-text-primary' : 'text-text-muted'}`}>
          {t('pricing.monthly')}
        </span>
        <button
          onClick={() => setAnnual(!annual)}
          className={`relative w-12 h-6 rounded-full transition-colors ${annual ? 'bg-accent' : 'bg-elevated'}`}
          aria-label={t('pricing.toggleLabel')}
        >
          <span
            className={`absolute top-0.5 left-0.5 w-5 h-5 bg-void rounded-full transition-transform ${annual ? 'translate-x-6' : ''}`}
          />
        </button>
        <span className={`text-xs ${annual ? 'text-text-primary' : 'text-text-muted'}`}>
          {t('pricing.annual')}
        </span>
        {annual && (
          <span className="text-[10px] font-bold bg-accent/15 text-accent border border-accent/25 rounded-full px-2.5 py-0.5">
            {t('pricing.save17')}
          </span>
        )}
      </div>

      {/* Pricing Cards */}
      <section className="px-6 pb-20">
        <div className="max-w-5xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-5">
          {TIERS.map((tier) => {
            const isFeatured = tier.featured
            const price = annual ? tier.annual : tier.monthly
            const unit = annual ? t('pricing.perBedYear') : t('pricing.perBedMonth')
            const isEnterprise = tier.key === 'enterprise'

            return (
              <div
                key={tier.key}
                className={`relative rounded-xl p-6 border flex flex-col ${
                  isFeatured
                    ? 'border-accent bg-surface glow-accent'
                    : 'border-border bg-surface'
                }`}
              >
                {isFeatured && (
                  <span className="absolute -top-3 left-1/2 -translate-x-1/2 text-[10px] font-bold tracking-widest uppercase bg-accent text-void rounded-full px-3 py-1">
                    {t('pricing.mostPopular')}
                  </span>
                )}
                <h3 className="font-heading text-lg font-bold mb-1">{t(`pricing.${tier.key}Name`)}</h3>
                <div className="mb-4">
                  <span className="font-heading text-3xl font-extrabold text-accent">${price}</span>
                  <span className="text-xs text-text-muted ml-1">{unit}</span>
                </div>
                <ul className="space-y-2.5 mb-6 flex-1">
                  {[1, 2, 3, 4].map((i) => (
                    <li key={i} className="flex items-start gap-2 text-xs text-text-secondary">
                      <Check className="w-3.5 h-3.5 text-accent mt-0.5 shrink-0" />
                      <span>{t(`pricing.${tier.key}Feat${i}`)}</span>
                    </li>
                  ))}
                </ul>
                {isEnterprise ? (
                  <a
                    href="mailto:sales@sepsisvitals.com"
                    className="inline-flex items-center justify-center gap-2 border border-border text-text-primary text-sm font-bold px-6 py-3 rounded hover:border-accent hover:text-accent transition-colors"
                  >
                    {t('pricing.contactSales')}
                  </a>
                ) : (
                  <Link
                    to="/login"
                    className={`inline-flex items-center justify-center gap-2 text-sm font-bold px-6 py-3 rounded transition-all ${
                      isFeatured
                        ? 'bg-accent text-void hover:bg-accent-dim hover:shadow-[0_0_30px_rgba(0,255,157,0.3)]'
                        : 'border border-border text-text-primary hover:border-accent hover:text-accent'
                    }`}
                  >
                    {t(`pricing.${tier.key}Cta`)}
                    <ArrowRight className="w-4 h-4" />
                  </Link>
                )}
              </div>
            )
          })}
        </div>
      </section>

      {/* Feature Comparison Matrix */}
      <section className="px-6 py-20 bg-surface">
        <div className="max-w-5xl mx-auto">
          <p className="text-[10px] font-bold tracking-[0.18em] uppercase text-accent mb-3">
            {t('pricing.comparisonLabel')}
          </p>
          <h2 className="font-heading text-2xl sm:text-3xl font-bold mb-8">
            {t('pricing.comparisonTitle')}
          </h2>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left py-3 pr-4 text-text-muted font-normal w-1/3">
                    {t('pricing.feature')}
                  </th>
                  {TIERS.map((tier) => (
                    <th key={tier.key} className="text-center py-3 px-4 font-heading font-semibold">
                      {t(`pricing.${tier.key}Name`)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {FEATURE_MATRIX_KEYS.map((fk) => (
                  <tr key={fk} className="border-b border-border/50">
                    <td className="py-3 pr-4 text-text-secondary">
                      {t(`pricing.matrix_${fk}`)}
                    </td>
                    {FEATURE_AVAILABILITY[fk].map((available, i) => (
                      <td key={i} className="text-center py-3 px-4">
                        {available ? (
                          <Check className="w-4 h-4 text-accent mx-auto" />
                        ) : (
                          <X className="w-4 h-4 text-text-muted mx-auto" />
                        )}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* ROI Calculator */}
      <section className="px-6 py-20">
        <div className="max-w-4xl mx-auto">
          <p className="text-[10px] font-bold tracking-[0.18em] uppercase text-accent mb-3">
            {t('pricing.roiLabel')}
          </p>
          <h2 className="font-heading text-2xl sm:text-3xl font-bold mb-3">
            {t('pricing.roiTitle')}
          </h2>
          <p className="text-sm text-text-secondary max-w-xl mb-10 leading-relaxed">
            {t('pricing.roiSub')}
          </p>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Inputs */}
            <div className="space-y-6">
              <div>
                <label className="block text-xs text-text-secondary mb-2">
                  {t('pricing.bedsLabel')}: <span className="text-accent font-bold">{beds}</span>
                </label>
                <input
                  type="range"
                  min={10}
                  max={2000}
                  step={10}
                  value={beds}
                  onChange={(e) => setBeds(Number(e.target.value))}
                  className="w-full accent-accent"
                />
                <div className="flex justify-between text-[10px] text-text-muted mt-1">
                  <span>10</span>
                  <span>2,000</span>
                </div>
              </div>
              <div>
                <label className="block text-xs text-text-secondary mb-2">
                  {t('pricing.casesLabel')}: <span className="text-accent font-bold">{sepsisCases}</span>
                </label>
                <input
                  type="range"
                  min={10}
                  max={1000}
                  step={5}
                  value={sepsisCases}
                  onChange={(e) => setSepsisCases(Number(e.target.value))}
                  className="w-full accent-accent"
                />
                <div className="flex justify-between text-[10px] text-text-muted mt-1">
                  <span>10</span>
                  <span>1,000</span>
                </div>
              </div>
              <p className="text-[10px] text-text-muted leading-relaxed">
                {t('pricing.roiDisclaimer')}
              </p>
            </div>

            {/* Outputs */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="bg-surface border border-border rounded-lg p-5">
                <p className="text-[10px] text-text-muted uppercase tracking-widest mb-1">
                  {t('pricing.annualSavings')}
                </p>
                <p className="font-heading text-2xl font-extrabold text-accent">
                  ${savings.toLocaleString()}
                </p>
              </div>
              <div className="bg-surface border border-border rounded-lg p-5">
                <p className="text-[10px] text-text-muted uppercase tracking-widest mb-1">
                  {t('pricing.livesSaved')}
                </p>
                <p className="font-heading text-2xl font-extrabold text-accent">
                  {livesSaved}
                </p>
              </div>
              <div className="bg-surface border border-border rounded-lg p-5">
                <p className="text-[10px] text-text-muted uppercase tracking-widest mb-1">
                  {t('pricing.annualCost')}
                </p>
                <p className="font-heading text-2xl font-extrabold text-text-primary">
                  ${cost.toLocaleString()}
                </p>
                <p className="text-[10px] text-text-muted mt-1">
                  {t('pricing.proTier')}
                </p>
              </div>
              <div className="bg-surface border border-accent/30 rounded-lg p-5 glow-accent">
                <p className="text-[10px] text-text-muted uppercase tracking-widest mb-1">
                  {t('pricing.roiPercent')}
                </p>
                <p className="font-heading text-2xl font-extrabold text-accent">
                  {roi.toLocaleString()}%
                </p>
              </div>
            </div>
          </div>
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
