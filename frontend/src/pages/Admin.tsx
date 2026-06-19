import { Settings, Shield, Database, Cpu, Globe } from 'lucide-react'

export function Admin() {
  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h1 className="font-heading text-2xl font-bold flex items-center gap-2">
          <Settings className="w-6 h-6 text-accent" />
          Admin
        </h1>
        <p className="text-sm text-text-secondary mt-1">System configuration and monitoring</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* System Status */}
        <div className="bg-surface border border-border rounded-lg p-5">
          <h2 className="font-heading text-sm font-semibold flex items-center gap-2 mb-4">
            <Cpu className="w-4 h-4 text-info" /> System Status
          </h2>
          <div className="space-y-3">
            {[
              { label: 'API Server', status: 'Running', ok: true },
              { label: 'ML Model', status: 'GradientBoosting v1.0', ok: true },
              { label: 'SQLite State', status: 'WAL Mode', ok: true },
              { label: 'WebSocket', status: 'Connected', ok: true },
              { label: 'FHIR Listener', status: 'Standby', ok: true },
            ].map((item) => (
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
            {[
              { label: 'Patient State', value: 'SQLite WAL' },
              { label: 'User Store', value: 'SQLite WAL' },
              { label: 'Model', value: 'GradientBoosting 380KB' },
              { label: 'Training Data', value: '15K patients synthetic' },
              { label: 'Test AUROC', value: '0.88' },
            ].map((item) => (
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
