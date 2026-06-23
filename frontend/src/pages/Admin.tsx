import { useEffect, useState } from 'react'
import { Settings, Shield, Database, Cpu, Globe } from 'lucide-react'
import { api, isDemo } from '../lib/api'

export function Admin() {
  const [systemStatus, setSystemStatus] = useState([
    { label: 'API Server', status: 'Unknown', ok: false },
    { label: 'ML Model', status: 'Loading...', ok: false },
    { label: 'Database', status: 'Loading...', ok: false },
    { label: 'WebSocket', status: 'Loading...', ok: false },
    { label: 'Redis', status: 'Loading...', ok: false },
  ])

  const [dataInfo, setDataInfo] = useState([
    { label: 'Patient State', value: 'SQLite WAL' },
    { label: 'User Store', value: 'SQLite WAL' },
    { label: 'Model', value: 'Loading...' },
    { label: 'Training Data', value: '20K patients synthetic' },
    { label: 'Test AUROC', value: 'Loading...' },
  ])

  useEffect(() => {
    if (isDemo) {
      setSystemStatus([
        { label: 'API Server', status: 'Demo Mode', ok: true },
        { label: 'ML Model', status: 'GradientBoosting v2.0', ok: true },
        { label: 'Database', status: 'N/A (Demo)', ok: true },
        { label: 'WebSocket', status: 'N/A (Demo)', ok: true },
        { label: 'Redis', status: 'N/A (Demo)', ok: true },
      ])
      setDataInfo([
        { label: 'Patient State', value: 'SQLite WAL' },
        { label: 'User Store', value: 'SQLite WAL' },
        { label: 'Model', value: 'GradientBoosting' },
        { label: 'Training Data', value: '20K patients synthetic' },
        { label: 'Test AUROC', value: '0.92' },
      ])
      return
    }

    api.systemHealth()
      .then((health) => {
        setSystemStatus([
          { label: 'API Server', status: `${health.status} (v${health.version})`, ok: health.status === 'healthy' },
          { label: 'ML Model', status: health.model_loaded ? 'Loaded' : 'Not loaded', ok: health.model_loaded },
          { label: 'Database', status: health.database === 'healthy' ? 'Connected' : health.database, ok: health.database === 'healthy' },
          { label: 'WebSocket', status: `${health.websocket_connections} conn`, ok: true },
          { label: 'Redis', status: health.redis === 'healthy' ? 'Connected' : health.redis, ok: health.redis === 'healthy' },
        ])
      })
      .catch(() => {
        setSystemStatus((prev) => prev.map((s) => ({ ...s, status: 'Unreachable', ok: false })))
      })

    api.modelInfo()
      .then((info) => {
        const auroc = info.metrics?.val_auroc ?? info.metrics?.test_auroc
        setDataInfo([
          { label: 'Patient State', value: 'SQLite WAL' },
          { label: 'User Store', value: 'SQLite WAL' },
          { label: 'Model', value: `${info.model_name} (${info.feature_count} features)` },
          { label: 'Calibrated', value: info.is_calibrated ? 'Yes (Platt)' : 'No' },
          { label: 'Test AUROC', value: auroc ? auroc.toFixed(4) : 'N/A' },
        ])
      })
      .catch(() => {})
  }, [])
  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h1 className="font-heading text-2xl font-bold flex items-center gap-2">
          <Settings className="w-6 h-6 text-accent" />
          Admin
        </h1>
        <p className="text-sm text-text-secondary mt-1">
          System configuration and monitoring
          {isDemo && <span className="ml-2 text-xs text-warning">(Demo Mode)</span>}
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* System Status */}
        <div className="bg-surface border border-border rounded-lg p-5">
          <h2 className="font-heading text-sm font-semibold flex items-center gap-2 mb-4">
            <Cpu className="w-4 h-4 text-info" /> System Status
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
            <Shield className="w-4 h-4 text-accent" /> Security
          </h2>
          <div className="space-y-3">
            {[
              { label: 'Auth', value: 'JWT + PBKDF2' },
              { label: 'Rate Limiting', value: '10/s API, 2/s ML, 1/s Billing' },
              { label: 'HSTS', value: '2yr + Preload' },
              { label: 'CSP', value: 'Strict' },
              { label: 'Injection Guard', value: '21 Patterns' },
              { label: 'LLM Copilot', value: 'Enterprise-only' },
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
            <Database className="w-4 h-4 text-warning" /> Data
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
            <Globe className="w-4 h-4 text-info" /> Integrations
          </h2>
          <div className="space-y-3">
            {[
              { label: 'HL7v2 MLLP', value: 'Ready (port 2575)' },
              { label: 'FHIR R4 Webhook', value: 'Ready' },
              { label: 'Stripe Billing', value: '3 Plans' },
              { label: 'Twilio SMS', value: 'Configured' },
              { label: 'Prometheus', value: '/metrics' },
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
