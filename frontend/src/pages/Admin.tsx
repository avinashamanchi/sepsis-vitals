import { useEffect, useState } from 'react'
import { Settings, Shield, Database, Cpu, Globe } from 'lucide-react'
import { api, isDemo } from '../lib/api'
import { useTranslation } from 'react-i18next'

export function Admin() {
  const { t } = useTranslation()

  const [systemStatus, setSystemStatus] = useState([
    { label: t('admin.apiServer'), status: t('common.unknown'), ok: false },
    { label: t('admin.mlModel'), status: t('common.loading'), ok: false },
    { label: t('admin.database'), status: t('common.loading'), ok: false },
    { label: t('admin.webSocket'), status: t('common.loading'), ok: false },
    { label: t('admin.redis'), status: t('common.loading'), ok: false },
  ])

  const [dataInfo, setDataInfo] = useState([
    { label: t('admin.patientState'), value: t('admin.sqliteWal') },
    { label: t('admin.userStore'), value: t('admin.sqliteWal') },
    { label: t('admin.model'), value: t('common.loading') },
    { label: t('admin.trainingData'), value: t('admin.syntheticData') },
    { label: t('admin.testAuroc'), value: t('common.loading') },
  ])

  useEffect(() => {
    if (isDemo) {
      setSystemStatus([
        { label: t('admin.apiServer'), status: t('admin.demoMode'), ok: true },
        { label: t('admin.mlModel'), status: 'GradientBoosting v2.0', ok: true },
        { label: t('admin.database'), status: t('admin.naDemo'), ok: true },
        { label: t('admin.webSocket'), status: t('admin.naDemo'), ok: true },
        { label: t('admin.redis'), status: t('admin.naDemo'), ok: true },
      ])
      setDataInfo([
        { label: t('admin.patientState'), value: t('admin.sqliteWal') },
        { label: t('admin.userStore'), value: t('admin.sqliteWal') },
        { label: t('admin.model'), value: 'GradientBoosting' },
        { label: t('admin.trainingData'), value: t('admin.syntheticData') },
        { label: t('admin.testAuroc'), value: '0.92' },
      ])
      return
    }

    api.systemHealth()
      .then((health) => {
        setSystemStatus([
          { label: t('admin.apiServer'), status: `${health.status} (v${health.version})`, ok: health.status === 'healthy' },
          { label: t('admin.mlModel'), status: health.model_loaded ? t('admin.loaded') : t('admin.notLoaded'), ok: health.model_loaded },
          { label: t('admin.database'), status: health.database === 'healthy' ? t('common.connected') : health.database, ok: health.database === 'healthy' },
          { label: t('admin.webSocket'), status: t('admin.connections', { n: health.websocket_connections }), ok: true },
          { label: t('admin.redis'), status: health.redis === 'healthy' ? t('common.connected') : health.redis, ok: health.redis === 'healthy' },
        ])
      })
      .catch(() => {
        setSystemStatus((prev) => prev.map((s) => ({ ...s, status: t('admin.unreachable'), ok: false })))
      })

    api.modelInfo()
      .then((info) => {
        const auroc = info.metrics?.val_auroc ?? info.metrics?.test_auroc
        setDataInfo([
          { label: t('admin.patientState'), value: t('admin.sqliteWal') },
          { label: t('admin.userStore'), value: t('admin.sqliteWal') },
          { label: t('admin.model'), value: `${info.model_name} (${info.feature_count} features)` },
          { label: t('admin.calibrated'), value: info.is_calibrated ? t('admin.yesPlatt') : t('admin.no') },
          { label: t('admin.testAuroc'), value: auroc ? auroc.toFixed(4) : t('common.na') },
        ])
      })
      .catch(() => {})
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h1 className="font-heading text-2xl font-bold flex items-center gap-2">
          <Settings className="w-6 h-6 text-accent" />
          {t('admin.title')}
        </h1>
        <p className="text-sm text-text-secondary mt-1">
          {t('admin.subtitle')}
          {isDemo && <span className="ml-2 text-xs text-warning">{t('common.demoMode')}</span>}
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* System Status */}
        <div className="bg-surface border border-border rounded-lg p-5">
          <h2 className="font-heading text-sm font-semibold flex items-center gap-2 mb-4">
            <Cpu className="w-4 h-4 text-info" /> {t('admin.systemStatus')}
          </h2>
          <div className="space-y-3">
            {systemStatus.map((item) => (
              <div key={item.label} className="flex items-center justify-between py-2 border-b border-border/50 last:border-0">
                <span className="text-sm text-text-secondary">{item.label}</span>
                <span className={`text-xs font-medium ${item.ok ? 'text-accent' : 'text-danger'}`}>
                  {item.status}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Security */}
        <div className="bg-surface border border-border rounded-lg p-5">
          <h2 className="font-heading text-sm font-semibold flex items-center gap-2 mb-4">
            <Shield className="w-4 h-4 text-accent" /> {t('admin.security')}
          </h2>
          <div className="space-y-3">
            {[
              { label: t('admin.auth'), value: t('admin.jwtConfigured') },
              { label: t('admin.rateLimiting'), value: isDemo ? t('admin.notVerifiedDemo') : t('admin.rateLimitValues') },
              { label: t('admin.securityHeaders'), value: isDemo ? t('admin.notVerifiedDemo') : t('admin.hstsCSP') },
              { label: t('admin.inputValidation'), value: t('admin.parameterizedQueries') },
              { label: t('admin.sessionTimeout'), value: t('admin.hipaaTimeout') },
              { label: t('admin.llmCopilot'), value: t('admin.enterpriseOnly') },
            ].map((item) => (
              <div key={item.label} className="flex items-center justify-between py-2 border-b border-border/50 last:border-0">
                <span className="text-sm text-text-secondary">{item.label}</span>
                <span className="text-xs text-text-primary">{item.value}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Database */}
        <div className="bg-surface border border-border rounded-lg p-5">
          <h2 className="font-heading text-sm font-semibold flex items-center gap-2 mb-4">
            <Database className="w-4 h-4 text-warning" /> {t('admin.data')}
          </h2>
          <div className="space-y-3">
            {dataInfo.map((item) => (
              <div key={item.label} className="flex items-center justify-between py-2 border-b border-border/50 last:border-0">
                <span className="text-sm text-text-secondary">{item.label}</span>
                <span className="text-xs text-text-primary">{item.value}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Integrations */}
        <div className="bg-surface border border-border rounded-lg p-5">
          <h2 className="font-heading text-sm font-semibold flex items-center gap-2 mb-4">
            <Globe className="w-4 h-4 text-info" /> {t('admin.integrations')}
          </h2>
          <div className="space-y-3">
            {[
              { label: t('admin.hl7v2'), value: isDemo ? t('admin.notDeployedDemo') : t('admin.port2575') },
              { label: t('admin.fhirR4'), value: isDemo ? t('admin.notDeployedDemo') : t('admin.webhookReady') },
              { label: t('admin.stripeBilling'), value: isDemo ? t('admin.notDeployedDemo') : t('admin.envConfigured') },
              { label: t('admin.smsAlerts'), value: isDemo ? t('admin.notDeployedDemo') : t('admin.envConfigured') },
              { label: t('admin.prometheus'), value: t('admin.metricsEndpoint') },
            ].map((item) => (
              <div key={item.label} className="flex items-center justify-between py-2 border-b border-border/50 last:border-0">
                <span className="text-sm text-text-secondary">{item.label}</span>
                <span className="text-xs text-text-primary">{item.value}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
