import { useState, useMemo } from 'react'
import { useTranslation } from 'react-i18next'
import { Globe, ChevronDown } from 'lucide-react'
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell, Legend,
} from 'recharts'
import {
  AGE_GROUPS, SEX, ETHNICITY,
  BP_SYSTOLIC, BP_DIASTOLIC, HEART_RATE, RESPIRATORY_RATE, SPO2,
  BP_TRENDS, BP_BY_ETHNICITY, HYPERTENSION_PREVALENCE,
  CYCLES,
  type Distribution, type BPTrend,
} from '../data/nhanes'

// ── Types ──

type VitalKey = 'bp' | 'hr' | 'rr' | 'spo2'

interface PercentileBarDatum {
  label: string
  value: number
  zone: 'normal' | 'borderline' | 'abnormal'
}

interface TrendDatum {
  cycle: string
  sbp?: number
  dbp?: number
  mean?: number
}

interface PrevalenceDatum {
  ageGroup: string
  prevalence: number
}

interface EthnicityDatum {
  ethnicity: string
  sbp_mean: number
  dbp_mean: number
}

// ── Constants ──

const VITAL_OPTIONS: { key: VitalKey; labelKey: string }[] = [
  { key: 'bp', labelKey: 'population.vitalBP' },
  { key: 'hr', labelKey: 'population.vitalHR' },
  { key: 'rr', labelKey: 'population.vitalRR' },
  { key: 'spo2', labelKey: 'population.vitalSpO2' },
]

const CHART_TOOLTIP = {
  contentStyle: {
    background: '#0a1120',
    border: '1px solid rgba(255,255,255,0.06)',
    borderRadius: 8,
    color: '#e8f4ff',
    fontSize: 12,
  },
}

const ZONE_COLORS: Record<string, string> = {
  normal: '#00ff9d',
  borderline: '#ffb830',
  abnormal: '#ff3b5c',
}

const LINE_COLORS = ['#38b4ff', '#ff6b35', '#00ff9d', '#a78bfa']

// ── Helpers ──

function getVitalTable(vital: VitalKey, sex: string, ageGroup: string): Distribution | null {
  const tables: Record<VitalKey, Record<string, Record<string, Distribution>>> = {
    bp: BP_SYSTOLIC,
    hr: HEART_RATE,
    rr: RESPIRATORY_RATE,
    spo2: SPO2,
  }
  return tables[vital]?.[sex]?.[ageGroup] ?? null
}

function getDBPDistribution(sex: string, ageGroup: string): Distribution | null {
  return BP_DIASTOLIC[sex]?.[ageGroup] ?? null
}

function classifyBPZone(percentileLabel: string, value: number): 'normal' | 'borderline' | 'abnormal' {
  // For SBP: normal < 120, borderline 120-139, abnormal >= 140
  if (percentileLabel === 'p5' || percentileLabel === 'p25') {
    if (value < 90) return 'abnormal'
    if (value < 100) return 'borderline'
    return 'normal'
  }
  if (value >= 140) return 'abnormal'
  if (value >= 120) return 'borderline'
  return 'normal'
}

function classifyHRZone(_label: string, value: number): 'normal' | 'borderline' | 'abnormal' {
  if (value < 50 || value > 130) return 'abnormal'
  if (value < 60 || value > 100) return 'borderline'
  return 'normal'
}

function classifyRRZone(_label: string, value: number): 'normal' | 'borderline' | 'abnormal' {
  if (value < 10 || value > 24) return 'abnormal'
  if (value < 12 || value > 20) return 'borderline'
  return 'normal'
}

function classifySpO2Zone(_label: string, value: number): 'normal' | 'borderline' | 'abnormal' {
  if (value < 90) return 'abnormal'
  if (value < 94) return 'borderline'
  return 'normal'
}

function classifyZone(vital: VitalKey, label: string, value: number): 'normal' | 'borderline' | 'abnormal' {
  switch (vital) {
    case 'bp': return classifyBPZone(label, value)
    case 'hr': return classifyHRZone(label, value)
    case 'rr': return classifyRRZone(label, value)
    case 'spo2': return classifySpO2Zone(label, value)
  }
}

function vitalUnit(vital: VitalKey): string {
  switch (vital) {
    case 'bp': return 'mmHg'
    case 'hr': return 'bpm'
    case 'rr': return '/min'
    case 'spo2': return '%'
  }
}

// ── Dropdown Component ──

function Dropdown({
  label,
  value,
  options,
  onChange,
}: {
  label: string
  value: string
  options: { value: string; label: string }[]
  onChange: (v: string) => void
}) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-xs text-text-muted font-medium uppercase tracking-wide">{label}</label>
      <div className="relative">
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="appearance-none bg-elevated border border-border rounded-md px-3 py-2 pr-8 text-sm text-text-primary focus:outline-none focus:ring-1 focus:ring-accent cursor-pointer w-full"
        >
          {options.map((o) => (
            <option key={o.value} value={o.value}>{o.label}</option>
          ))}
        </select>
        <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted pointer-events-none" />
      </div>
    </div>
  )
}

// ── Toggle Component ──

function Toggle({
  label,
  options,
  value,
  onChange,
}: {
  label: string
  options: { value: string; label: string }[]
  value: string
  onChange: (v: string) => void
}) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-xs text-text-muted font-medium uppercase tracking-wide">{label}</span>
      <div className="flex bg-elevated border border-border rounded-md overflow-hidden">
        {options.map((o) => (
          <button
            key={o.value}
            onClick={() => onChange(o.value)}
            className={`px-4 py-2 text-sm font-medium transition-colors ${
              value === o.value
                ? 'bg-accent/20 text-accent'
                : 'text-text-secondary hover:text-text-primary'
            }`}
          >
            {o.label}
          </button>
        ))}
      </div>
    </div>
  )
}

// ── Section Wrapper ──

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-surface border border-border rounded-lg">
      <div className="px-4 py-3 border-b border-border">
        <h2 className="font-heading text-sm font-semibold">{title}</h2>
      </div>
      <div className="p-4">
        {children}
      </div>
    </div>
  )
}

// ── Main Component ──

export function Population() {
  const { t } = useTranslation()

  // Filter state
  const [ageGroup, setAgeGroup] = useState('40-49')
  const [sex, setSex] = useState('Male')
  const [ethnicity, setEthnicity] = useState('All')
  const [vital, setVital] = useState<VitalKey>('bp')

  // ── Distribution Data ──
  const distributionData = useMemo((): PercentileBarDatum[] => {
    const dist = getVitalTable(vital, sex, ageGroup)
    if (!dist) return []
    const keys = ['p5', 'p25', 'p50', 'p75', 'p95'] as const
    return keys.map((k) => ({
      label: k.toUpperCase(),
      value: dist[k],
      zone: classifyZone(vital, k, dist[k]),
    }))
  }, [vital, sex, ageGroup])

  const dbpDistributionData = useMemo((): PercentileBarDatum[] | null => {
    if (vital !== 'bp') return null
    const dist = getDBPDistribution(sex, ageGroup)
    if (!dist) return null
    const keys = ['p5', 'p25', 'p50', 'p75', 'p95'] as const
    return keys.map((k) => ({
      label: k.toUpperCase(),
      value: dist[k],
      zone: classifyZone(vital, k, dist[k]),
    }))
  }, [vital, sex, ageGroup])

  // ── Trend Data ──
  const trendData = useMemo((): TrendDatum[] => {
    if (vital === 'bp') {
      return CYCLES.map((cycle) => {
        const d = BP_TRENDS[cycle] as BPTrend | undefined
        return {
          cycle: cycle.replace('20', "'").replace('-20', "-'").replace("-'0", "-'0"),
          sbp: d?.sbp,
          dbp: d?.dbp,
        }
      })
    }
    // For other vitals, compute mean across age groups per cycle
    // NHANES data is cross-sectional, not longitudinal per cycle for HR/RR/SpO2
    // Show mean by age group as the "trend" instead
    return AGE_GROUPS.map((ag) => {
      const dist = getVitalTable(vital, sex, ag)
      return {
        cycle: ag,
        mean: dist?.mean,
      }
    })
  }, [vital, sex])

  // ── Reference Ranges Table ──
  const referenceData = useMemo(() => {
    return AGE_GROUPS.map((ag) => {
      const dist = getVitalTable(vital, sex, ag)
      if (!dist) return { ageGroup: ag, mean: 0, sd: 0, p5: 0, p25: 0, p50: 0, p75: 0, p95: 0 }
      return { ageGroup: ag, ...dist }
    })
  }, [vital, sex])

  // ── Prevalence Data ──
  const prevalenceData = useMemo((): PrevalenceDatum[] => {
    return AGE_GROUPS.map((ag) => ({
      ageGroup: ag,
      prevalence: Math.round((HYPERTENSION_PREVALENCE[ag] ?? 0) * 100),
    }))
  }, [])

  // ── Ethnicity Comparison ──
  const ethnicityData = useMemo((): EthnicityDatum[] => {
    return ETHNICITY.map((eth) => {
      const d = BP_BY_ETHNICITY[eth]
      return {
        ethnicity: eth.replace('Non-Hispanic ', 'NH '),
        sbp_mean: d?.sbp_mean ?? 0,
        dbp_mean: d?.dbp_mean ?? 0,
      }
    })
  }, [])

  // ── Vital label for display ──
  const vitalLabel = VITAL_OPTIONS.find((v) => v.key === vital)?.labelKey ?? ''

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div>
        <h1 className="font-heading text-2xl font-bold flex items-center gap-2">
          <Globe className="w-6 h-6 text-info" />
          {t('population.title')}
        </h1>
        <p className="text-sm text-text-secondary mt-1">
          {t('population.subtitle')}
        </p>
      </div>

      {/* Filter Bar */}
      <div className="sticky top-0 z-10 bg-background/80 backdrop-blur-md border border-border rounded-lg p-4">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Dropdown
            label={t('population.ageGroup')}
            value={ageGroup}
            options={AGE_GROUPS.map((ag) => ({ value: ag, label: ag }))}
            onChange={setAgeGroup}
          />
          <Toggle
            label={t('population.sex')}
            options={SEX.map((s) => ({ value: s, label: s }))}
            value={sex}
            onChange={setSex}
          />
          <Dropdown
            label={t('population.ethnicity')}
            value={ethnicity}
            options={[
              { value: 'All', label: t('common.all') },
              ...ETHNICITY.map((e) => ({ value: e, label: e })),
            ]}
            onChange={setEthnicity}
          />
          <Dropdown
            label={t('population.vitalSign')}
            value={vital}
            options={VITAL_OPTIONS.map((v) => ({ value: v.key, label: t(v.labelKey) }))}
            onChange={(v) => setVital(v as VitalKey)}
          />
        </div>
      </div>

      {/* Distribution Section */}
      <Section title={`${t('population.distribution')} — ${t(vitalLabel)} (${sex}, ${ageGroup})`}>
        <div className={vital === 'bp' ? 'grid grid-cols-1 lg:grid-cols-2 gap-6' : ''}>
          {/* SBP or primary vital */}
          <div>
            {vital === 'bp' && (
              <h3 className="text-xs text-text-muted uppercase tracking-wide mb-2">{t('population.systolic')}</h3>
            )}
            <div className="h-[240px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={distributionData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis
                    dataKey="label"
                    stroke="#4a6080"
                    tick={{ fill: '#4a6080', fontSize: 11 }}
                    tickLine={false}
                  />
                  <YAxis
                    stroke="#4a6080"
                    tick={{ fill: '#4a6080', fontSize: 11 }}
                    tickLine={false}
                    axisLine={false}
                    unit={` ${vitalUnit(vital)}`}
                  />
                  <Tooltip
                    {...CHART_TOOLTIP}
                    formatter={(value: number) => [`${value} ${vitalUnit(vital)}`, t('population.value')]}
                  />
                  <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                    {distributionData.map((entry, i) => (
                      <Cell key={i} fill={ZONE_COLORS[entry.zone]} fillOpacity={0.85} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* DBP (only for BP) */}
          {vital === 'bp' && dbpDistributionData && (
            <div>
              <h3 className="text-xs text-text-muted uppercase tracking-wide mb-2">{t('population.diastolic')}</h3>
              <div className="h-[240px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={dbpDistributionData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                    <XAxis
                      dataKey="label"
                      stroke="#4a6080"
                      tick={{ fill: '#4a6080', fontSize: 11 }}
                      tickLine={false}
                    />
                    <YAxis
                      stroke="#4a6080"
                      tick={{ fill: '#4a6080', fontSize: 11 }}
                      tickLine={false}
                      axisLine={false}
                      unit=" mmHg"
                    />
                    <Tooltip
                      {...CHART_TOOLTIP}
                      formatter={(value: number) => [`${value} mmHg`, t('population.value')]}
                    />
                    <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                      {dbpDistributionData.map((entry, i) => (
                        <Cell key={i} fill={ZONE_COLORS[entry.zone]} fillOpacity={0.85} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </div>

        {/* Legend */}
        <div className="flex items-center gap-6 mt-4 text-xs text-text-muted">
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-sm" style={{ background: ZONE_COLORS.normal }} />
            {t('population.zoneNormal')}
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-sm" style={{ background: ZONE_COLORS.borderline }} />
            {t('population.zoneBorderline')}
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-sm" style={{ background: ZONE_COLORS.abnormal }} />
            {t('population.zoneAbnormal')}
          </span>
        </div>
      </Section>

      {/* Trends Section */}
      <Section title={vital === 'bp' ? t('population.trendsTitleBP') : t('population.trendsTitleByAge', { vital: t(vitalLabel) })}>
        <div className="h-[280px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={trendData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
              <XAxis
                dataKey="cycle"
                stroke="#4a6080"
                tick={{ fill: '#4a6080', fontSize: 11 }}
                tickLine={false}
              />
              <YAxis
                stroke="#4a6080"
                tick={{ fill: '#4a6080', fontSize: 11 }}
                tickLine={false}
                axisLine={false}
                domain={['dataMin - 2', 'dataMax + 2']}
              />
              <Tooltip {...CHART_TOOLTIP} />
              <Legend
                wrapperStyle={{ fontSize: 12, color: '#4a6080' }}
              />
              {vital === 'bp' ? (
                <>
                  <Line
                    type="monotone"
                    dataKey="sbp"
                    name={t('population.systolic')}
                    stroke={LINE_COLORS[0]}
                    strokeWidth={2}
                    dot={{ fill: LINE_COLORS[0], r: 3 }}
                  />
                  <Line
                    type="monotone"
                    dataKey="dbp"
                    name={t('population.diastolic')}
                    stroke={LINE_COLORS[1]}
                    strokeWidth={2}
                    dot={{ fill: LINE_COLORS[1], r: 3 }}
                  />
                </>
              ) : (
                <Line
                  type="monotone"
                  dataKey="mean"
                  name={t('population.mean')}
                  stroke={LINE_COLORS[0]}
                  strokeWidth={2}
                  dot={{ fill: LINE_COLORS[0], r: 3 }}
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Section>

      {/* Reference Ranges Table */}
      <Section title={`${t('population.referenceRanges')} — ${t(vitalLabel)} (${sex})`}>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <caption className="sr-only">{t('population.referenceRanges')}</caption>
            <thead>
              <tr className="text-text-muted text-xs uppercase border-b border-border">
                <th className="text-left px-4 py-3 font-medium">{t('population.colAgeGroup')}</th>
                <th className="text-right px-4 py-3 font-medium">{t('population.colMean')}</th>
                <th className="text-right px-4 py-3 font-medium">{t('population.colSD')}</th>
                <th className="text-right px-4 py-3 font-medium">P5</th>
                <th className="text-right px-4 py-3 font-medium">P25</th>
                <th className="text-right px-4 py-3 font-medium">P50</th>
                <th className="text-right px-4 py-3 font-medium">P75</th>
                <th className="text-right px-4 py-3 font-medium">P95</th>
              </tr>
            </thead>
            <tbody>
              {referenceData.map((row) => (
                <tr
                  key={row.ageGroup}
                  className={`border-b border-border/50 transition-colors ${
                    row.ageGroup === ageGroup
                      ? 'bg-accent/10 text-text-primary'
                      : 'text-text-secondary hover:bg-elevated/50'
                  }`}
                >
                  <td className="px-4 py-3 font-medium">{row.ageGroup}</td>
                  <td className="px-4 py-3 text-right tabular-nums">{row.mean.toFixed(1)}</td>
                  <td className="px-4 py-3 text-right tabular-nums">{row.sd.toFixed(1)}</td>
                  <td className="px-4 py-3 text-right tabular-nums">{row.p5}</td>
                  <td className="px-4 py-3 text-right tabular-nums">{row.p25}</td>
                  <td className="px-4 py-3 text-right tabular-nums">{row.p50}</td>
                  <td className="px-4 py-3 text-right tabular-nums">{row.p75}</td>
                  <td className="px-4 py-3 text-right tabular-nums">{row.p95}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      {/* Prevalence Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Hypertension Prevalence by Age */}
        <Section title={t('population.hypertensionPrevalence')}>
          <div className="h-[280px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={prevalenceData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis
                  dataKey="ageGroup"
                  stroke="#4a6080"
                  tick={{ fill: '#4a6080', fontSize: 11 }}
                  tickLine={false}
                />
                <YAxis
                  stroke="#4a6080"
                  tick={{ fill: '#4a6080', fontSize: 11 }}
                  tickLine={false}
                  axisLine={false}
                  unit="%"
                />
                <Tooltip
                  {...CHART_TOOLTIP}
                  formatter={(value: number) => [`${value}%`, t('population.prevalence')]}
                />
                <Bar dataKey="prevalence" radius={[4, 4, 0, 0]}>
                  {prevalenceData.map((entry, i) => (
                    <Cell
                      key={i}
                      fill={
                        entry.prevalence >= 50
                          ? ZONE_COLORS.abnormal
                          : entry.prevalence >= 20
                            ? ZONE_COLORS.borderline
                            : ZONE_COLORS.normal
                      }
                      fillOpacity={0.85}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Section>

        {/* Ethnicity Comparison */}
        <Section title={t('population.ethnicityComparison')}>
          <div className="h-[280px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={ethnicityData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis
                  dataKey="ethnicity"
                  stroke="#4a6080"
                  tick={{ fill: '#4a6080', fontSize: 10 }}
                  tickLine={false}
                  interval={0}
                  angle={-20}
                  textAnchor="end"
                  height={50}
                />
                <YAxis
                  stroke="#4a6080"
                  tick={{ fill: '#4a6080', fontSize: 11 }}
                  tickLine={false}
                  axisLine={false}
                  unit=" mmHg"
                  domain={[60, 140]}
                />
                <Tooltip {...CHART_TOOLTIP} />
                <Legend wrapperStyle={{ fontSize: 12, color: '#4a6080' }} />
                <Bar
                  dataKey="sbp_mean"
                  name={t('population.systolic')}
                  fill={LINE_COLORS[0]}
                  radius={[4, 4, 0, 0]}
                  opacity={0.85}
                />
                <Bar
                  dataKey="dbp_mean"
                  name={t('population.diastolic')}
                  fill={LINE_COLORS[1]}
                  radius={[4, 4, 0, 0]}
                  opacity={0.85}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Section>
      </div>

      {/* Source attribution */}
      <p className="text-xs text-text-muted text-center pb-4">
        {t('population.source')}
      </p>
    </div>
  )
}
