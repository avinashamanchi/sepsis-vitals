# Research UI Expansion Notes

This note records the second layer added to the static site after the initial
roadmap dashboard.

## New UI Modules

1. Evidence map
   - Connects each major research source to a product surface.
   - Keeps the dashboard honest: burden, clinical guidelines, low-resource
     baselines, pediatric scope, and external validation risk are visible.

2. Score comparator lab
   - Simulates qSOFA, partial SIRS, NEWS2-style bedside acuity, and UVA-style
     LMIC baseline scores across a few hard-coded patient scenarios.
   - Purpose: show why the ML model needs to beat practical rules at a useful
     alert burden, not just report a flattering AUROC.

3. Site readiness calculator
   - Estimates usable/adjudicated episodes and clinician review hours from
     available episodes, missingness, and review minutes per case.
   - Purpose: make Phase 1 dataset risk concrete before partner sites commit.

4. Prospective validation monitor
   - Shows sensitivity, specificity, PPV, and alerts per 100 encounters for
     sensitive, balanced, and strict thresholds.
   - Purpose: put alert fatigue beside accuracy during threshold tuning.

5. Implementation backlog
   - Converts research needs into future app modules:
     data intake, adjudication, model evaluation, offline deployment, workflow
     simulation, and model-card generation.

## Evidence Anchors

- WHO sepsis fact sheet: burden, vulnerable groups, and early treatment.
- Surviving Sepsis Campaign adult guidelines 2026: immediate care and
  resuscitation framing.
- Phoenix pediatric sepsis criteria: pediatric definition and limits of using
  Phoenix for early screening.
- NEWS2: vital-sign early-warning comparator.
- UVA score: sub-Saharan Africa mortality baseline using accessible features.
- Epic Sepsis Model external validation: external validation and alert fatigue
  cautionary example.

## Design Constraint

The site remains static and GitHub Pages friendly. No external JS/CSS/CDN assets
are required, and all interactive behavior is plain browser JavaScript.
