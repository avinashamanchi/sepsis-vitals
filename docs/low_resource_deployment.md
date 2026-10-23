# Running Sepsis Vitals on one server in a low-resource hospital

**Status: research prototype. Not approved for clinical use.** Nothing here
makes it safe to base patient care on its output. Use it for training,
demonstration and, with the approvals in `docs/pilot/README.md`, a
silent-mode study. Never enter real patient data without the project
owner's written approval, a data-use agreement and ethics approval.

## What staff get

| Output | Basis | Use |
|---|---|---|
| **UVA score** (0–13; low 0–1, medium 2–4, high > 4) | Published and validated for in-hospital mortality in adult inpatients in six sub-Saharan African countries (Moore et al., BMJ Glob Health 2017, doi:10.1136/bmjgh-2017-000344) | A published triage aid. Its local performance has not been checked. |
| qSOFA, NEWS2-style, SIRS, shock index | Published definitions, with the gaps listed in each response | Published early-warning scores |
| Model output | A vital-signs-only model trained on **synthetic** data (`docs/pilot/model_card.md`) | Research only. Hide it from staff with silent mode (below). |

Everything works without labs, without internet, and without a pulse
oximeter (missing items are listed, never silently scored as normal).

## 1. Hardware and software

- One computer on the ward network that stays on:
  - 4 GB RAM and 2 CPU cores are enough for one ward;
  - 20 GB free disk;
  - a UPS is strongly advised.
- Linux (Ubuntu LTS) or Windows with Docker Desktop.
- Docker Engine 24+ and Docker Compose v2.
- Internet only for the first install (images and packages). After that it runs offline.

## 2. Install

```bash
git clone https://github.com/avinashamanchi/sepsis-vitals.git
```

```bash
cd sepsis-vitals && python3 scripts/setup_site.py --app-url http://WARD-SERVER-ADDRESS:8000
```

`setup_site.py` writes `docker/.env` with every secret, readable only by
its owner. **Copy that file to an encrypted USB drive and lock it away.**
Without it, encrypted patient identifiers and backups cannot be read. The
script never overwrites an existing `.env`.

## 3. Start

```bash
cd docker && docker compose up -d --build --wait
```

This starts:
- the database;
- Redis;
- the API;
- the dashboard on port 8000;
- nightly encrypted backups into `docker/backups`.

Optional services:
- `--profile monitoring` adds Prometheus and Grafana;
- `--profile offsite` adds a cloud copy of backups.

All services restart by themselves after a power cut.

Check it:
- `http://WARD-SERVER-ADDRESS:8000` opens the dashboard;
- `docker compose exec api python -m sepsis_vitals.audit_records verify` reports 0 altered records.

## 4. Accounts

Create the first administrator (the password is asked for, never typed on the command line):

```bash
docker compose exec api python -m sepsis_vitals.auth.admin_cli create-user --email admin@hospital.example --role system_admin
```

Then create ward staff:

```bash
docker compose exec api python -m sepsis_vitals.auth.admin_cli create-user --email nurse1@hospital.example --role nurse --site WARD-A
```

Turn on two-factor sign-in for administrators once they have enrolled. Check
enrolment first:

```bash
docker compose exec api python -m sepsis_vitals.auth.admin_cli mfa-status --roles system_admin
```

Then set `SEPSIS_MFA_REQUIRED_ROLES=system_admin` in `docker/.env` and restart.

## 5. Silent mode (for a study)

Add `SEPSIS_MODEL_DISPLAY=silent` to `docker/.env` and restart.

- Every observation is still scored and recorded, sealed and append-only.
- Staff see only the published rule-based scores. The model output is withheld everywhere: dashboard, monitor, alerts and API.
- An unrecognised value is treated as silent.

### Recording outcomes and evaluating

1. After discharge, a researcher records each patient's adjudicated outcome under the site's reference standard (`POST /study/outcomes`: sepsis yes or no, optional onset time and in-hospital death, and the definition's name and version). There is no free text. A correction is a new entry.
2. Progress: `GET /study/summary` (counts only).
3. Evaluate, locally and only under the approved analysis plan:

```bash
docker compose exec api python -m sepsis_vitals.study --site WARD-A --output-dir /app/state/local-study-report
```

The output is aggregates only: AUROC and AUPRC with confidence intervals for the model, UVA, NEWS2-style and qSOFA; alerts before onset; lead time. It contains no patient identifiers. Keep it outside the code repository.

## 6. Daily use

1. A nurse records vital signs per patient on each round, on any phone or PC on the ward network.
2. The scores appear immediately. Items not measured are listed.
3. Alerts follow the rule-based scores. Who responds to them, and how, must be agreed by your clinical lead (`docs/pilot/clinical_requirements.md`, R5).

Available languages: English, French, Portuguese, Swahili, Amharic, Arabic.

## 7. Backups and restore

Nightly encrypted dumps go to `docker/backups`, kept 7 days, 4 weeks and 6
months. Copy them to an encrypted external drive weekly.

**Test a restore before relying on it** (runbook §5). A restore drill is
a pilot gate.

## 8. Updating

```bash
git pull && cd docker && docker compose up -d --build --wait
```

Database migrations run automatically before the API starts. Rollback
steps are in `docs/pilot/runbook.md`.

## 9. Do not

- Do not expose ports other than 8000 to the network. The compose file binds the database and API to the local machine.
- Do not change `SEPSIS_PII_KEY` in `.env` by hand; use `docs/pii_key_rotation.md`.
- Do not show the model output to staff before a local validation study.
- Do not enter real patient data before the approvals above.
