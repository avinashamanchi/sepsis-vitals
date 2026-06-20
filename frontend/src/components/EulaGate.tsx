import { useState, useCallback } from 'react'
import { Shield, AlertTriangle, Scale, FileCheck } from 'lucide-react'

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
            Sepsis Vitals
          </h1>
          <p className="text-sm text-text-secondary mt-1">
            End User License Agreement
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
              <p className="font-semibold text-danger text-sm">Critical Notice</p>
              <p className="text-text-secondary mt-1">
                Sepsis Vitals is a <strong className="text-text-primary">research-grade investigational tool</strong>.
                It is <strong className="text-text-primary">NOT cleared, approved, or authorized</strong> by the
                U.S. Food and Drug Administration (FDA) or any other regulatory body for
                autonomous clinical decision-making, diagnosis, or treatment guidance.
              </p>
            </div>
          </div>

          <section>
            <h3 className="text-text-primary font-semibold flex items-center gap-2 mb-2">
              <FileCheck className="w-4 h-4 text-accent" />
              1. Intended Use
            </h3>
            <p>
              Sepsis Vitals ("the Software") is designed solely for <strong className="text-text-primary">research,
              educational, and investigational purposes</strong>. The Software generates sepsis risk
              predictions using machine learning models trained on synthetic data. These
              predictions are probabilistic estimates and are not a substitute for professional
              clinical judgment, laboratory results, or established diagnostic criteria.
            </p>
          </section>

          <section>
            <h3 className="text-text-primary font-semibold flex items-center gap-2 mb-2">
              <Scale className="w-4 h-4 text-accent" />
              2. No Medical Advice; Physician Responsibility
            </h3>
            <p>
              The Software does not provide medical advice. No output, score, alert, or
              recommendation produced by the Software should be used as the sole basis for any
              clinical decision, including but not limited to diagnosis, treatment initiation,
              medication administration, or patient triage.
            </p>
            <p className="mt-2">
              <strong className="text-text-primary">All clinical decisions and medical liability
              rest solely with the attending physician</strong> or qualified healthcare professional.
              The Software is intended to supplement, not replace, the clinical judgment
              of a licensed medical practitioner.
            </p>
          </section>

          <section>
            <h3 className="text-text-primary font-semibold mb-2">3. Regulatory Status</h3>
            <p>
              The Software has not undergone FDA 510(k) clearance, De Novo classification,
              Premarket Approval (PMA), or CE marking. It is not registered as a medical device
              in any jurisdiction. The Software is classified as a <strong className="text-text-primary">Research
              Use Only (RUO)</strong> tool and must not be deployed in clinical care pathways without
              the appropriate regulatory clearances.
            </p>
          </section>

          <section>
            <h3 className="text-text-primary font-semibold mb-2">4. Model Limitations</h3>
            <p>
              The current model (v2.0.0) is trained on NHANES-calibrated synthetic data.
              Performance metrics (including AUROC, sensitivity, specificity, and PPV) are
              pre-validation estimates and <strong className="text-text-primary">may not generalize</strong> to
              real clinical populations, electronic health record systems, or care settings
              outside the training distribution. A demographic fairness audit has not been
              completed. Model outputs may exhibit bias across age, sex, race, or ethnicity
              subgroups.
            </p>
          </section>

          <section>
            <h3 className="text-text-primary font-semibold mb-2">5. Limitation of Liability</h3>
            <p>
              TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THE SOFTWARE IS PROVIDED
              "AS IS" WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
              LIMITED TO WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE,
              OR NON-INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS, CONTRIBUTORS, OR
              COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY,
              WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF,
              OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
            </p>
          </section>

          <section>
            <h3 className="text-text-primary font-semibold mb-2">6. Data Privacy</h3>
            <p>
              Any patient data entered into the Software in this demonstration mode is
              processed locally in the browser and is not transmitted to external servers.
              In production deployments, the Software is designed for HIPAA-compliant
              infrastructure with SOC 2 Type II controls. Users are responsible for
              ensuring compliance with all applicable data protection regulations in
              their jurisdiction.
            </p>
          </section>

          <section>
            <h3 className="text-text-primary font-semibold mb-2">7. Acceptance</h3>
            <p>
              By clicking "I Agree" below, you acknowledge that you have read, understood,
              and agree to be bound by the terms of this agreement. If you do not agree,
              you may not access or use the Software.
            </p>
          </section>

          <p className="text-text-muted text-xs pt-2 border-t border-border">
            EULA Version {EULA_VERSION} &mdash; Effective June 2026 &mdash; Sepsis Vitals, Inc.
          </p>
        </div>

        {/* Acceptance Controls */}
        <div className="mt-5 space-y-4">
          {!scrolledToBottom && (
            <p className="text-xs text-text-muted text-center">
              Scroll to the bottom of the agreement to continue
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
              I have read and understand that Sepsis Vitals is a <strong className="text-text-primary">research-grade
              tool</strong>, is <strong className="text-text-primary">not FDA-cleared</strong>, and that all medical
              liability rests with the attending physician.
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
              I Agree &mdash; Enter Application
            </button>
          </div>

          <p className="text-[11px] text-text-muted text-center">
            This agreement is required each time the EULA version is updated.
          </p>
        </div>
      </div>
    </div>
  )
}
