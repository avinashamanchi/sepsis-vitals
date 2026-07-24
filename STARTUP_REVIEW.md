# Sepsis Vitals: ruthless startup review

## Verdict

The original project looked more mature than it was. It had a broad feature
surface—billing, six languages, FHIR, alerting, dashboards, infrastructure, and
compliance templates—but no external clinical validation. In medical software,
that is backwards. The product was selling certainty before it had earned
trust.

The company should not launch as “AI that detects sepsis hours earlier.” It
should launch as a **paid validation and silent-mode platform** that helps
hospitals determine whether a sepsis early-warning signal is safe and useful in
their population and workflow.

That positioning is narrower, more honest, easier to buy as a first engagement,
and creates the evidence required for a real clinical product.

## Keep

### 1. The workflow and evaluation infrastructure

FHIR/HL7 ingestion, data-quality checks, versioned predictions, audit trails,
calibration, subgroup analysis, drift monitoring, and alert-burden measurement
are the foundation of a credible company. They solve hard work that every
clinical-model deployment needs.

### 2. The ward-review interface

The prioritized queue, patient trends, monitoring view, and traceable model
drivers are useful for human-factors testing. Keep them, but label them as
research outputs until the relevant studies are complete.

### 3. Multi-language and constrained-network support

These capabilities match the stated interest in resource-constrained settings.
They only become an advantage after the product has real partnerships in those
settings, so maintain them without making them the center of the pitch.

### 4. Study and risk-management scaffolding

The IRB, adjudication, QMS, and data-governance drafts are useful starting
points. They are templates, not certifications or completed regulatory work.

## Cut or freeze

### 1. Unsupported outcome claims

The original homepage claimed earlier detection, mortality reduction, and 99%
specificity. The pricing page calculated “lives saved” and ROI from fixed
assumptions. None of that was established by this product. Those claims had to
go.

### 2. Self-serve per-bed pricing

Per-bed SaaS pricing before validation signals that the company does not
understand enterprise clinical adoption. The first commercial offer should be
a scoped, paid validation engagement with clear deliverables and a go/revise/
stop decision.

### 3. “HIPAA compliant” and “SOC 2 ready”

Code controls do not make an organization compliant. Remove the badges until
independent audits, policies, training, vendor management, incident response,
access reviews, and evidence collection exist.

### 4. Billing and AI-copilot expansion

Freeze both. Billing is not the bottleneck. A generative clinical copilot
creates a second, harder safety and validation problem. Do not spend founder
time there before one narrow workflow has evidence and demand.

### 5. Feature accumulation

Stop adding pages because competitors have them. Every feature must improve one
of four pilot outcomes: data readiness, evaluation validity, workflow fit, or
safety monitoring.

## What was fixed in this pass

- Rebuilt the public site around evidence, pilot fit, and explicit limitations.
- Removed the fabricated pricing and ROI funnel.
- Added a public evidence ledger and validation-gate roadmap.
- Added a structured founding-partner pilot page.
- Moved the legal gate to the research application; public visitors can now see
  the site without accepting an EULA.
- Replaced a broken Firebase/backend authentication split with the backend’s
  own JWT flow.
- Disabled open production registration unless explicitly enabled.
- Moved browser tokens to session storage.
- Removed bearer tokens from WebSocket URLs and access logs.
- Stopped service-worker caching of API responses and purged the legacy cache.
- Removed the unused, unencrypted browser outbox and its legacy IndexedDB data.
- Removed unvalidated treatment instructions from model recommendations.
- Froze the clinical copilot by default and removed treatment instructions from
  its fallback path.
- Removed the unused counterfactual endpoint, which could be mistaken for
  treatment simulation despite having no clinical validation.
- Removed the synthetic “time to critical” forecast from the product surface.
- Removed the treatment-bundle UI and disabled its backend router by default.
- Disabled the billing router by default so pilot learning, not checkout code,
  remains the commercial focus.
- Removed fabricated “99% specificity” UI math and uses server-produced scores.
- Labeled synthetic analytics and model metrics honestly.
- Removed the unreproducible “NHANES population explorer,” which presented
  hand-authored values as derived public-health estimates.
- Added frontend checks to CI and aligned package versions and setup docs.
- Repaired the backend typecheck, current scikit-learn calibration path, score
  response contract, and network-listener defaults.

## The next 90 days

### Days 1–30: secure one validation partner

- Recruit a clinical safety lead and regulatory advisor.
- Write one precise intended-use statement and one target population.
- Create a data dictionary and minimum viable cohort specification.
- Sign a paid evaluation statement of work with explicit negative-result terms.
- Freeze the model and statistical analysis plan before looking at outcomes.

### Days 31–60: run retrospective external validation

- Validate data provenance, missingness, temporal alignment, leakage, and label
  quality.
- Evaluate AUROC/AUPRC, calibration, sensitivity and workload at prespecified
  operating points, and performance across age, sex, site, and relevant
  subgroups.
- Perform error review with clinicians. Document where the model fails and why.
- Decide whether to revise the intended use, revise the model, or stop.

### Days 61–90: prepare silent mode

- Run human-factors sessions on the review queue and explanations.
- Define alert caps, downtime behavior, escalation ownership, and rollback.
- Complete a prospective protocol that does not influence care.
- Instrument alerts per patient-day, time-to-review, false-alert clusters, data
  outages, and clinician disagreement.

## Non-negotiable launch gates

Do not enable live clinical alerts until all are true:

1. An intended use and target population are frozen.
2. Independent external validation meets prespecified criteria.
3. Calibration and subgroup performance are acceptable.
4. Alert burden and failure modes are acceptable in silent mode.
5. Human-factors testing shows clinicians understand outputs and limitations.
6. The regulatory pathway has been assessed by qualified counsel.
7. Clinical safety, security, privacy, quality, and incident-response owners are
   named and operational.
8. A deployment can be stopped safely without disrupting standard care.

## The metric that matters now

Not users. Not beds. Not synthetic AUROC.

The company’s next meaningful milestone is: **one independent hospital dataset,
one locked evaluation, and one publishable result that survives clinical
scrutiny.**
