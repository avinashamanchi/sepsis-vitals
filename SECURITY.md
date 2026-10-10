# Security policy

Sepsis Vitals is investigational research software. It is not approved for
clinical use, and no deployment should process real patient data without
the approvals described in `compliance/intended_use_and_validation_plan.md`.

## Reporting a vulnerability

**Do not put vulnerability details, exploit code, credentials or any
patient data in a public issue, pull request or discussion.**

- **Preferred: GitHub private vulnerability reporting.** On the repository's
  **Security** tab, choose **Report a vulnerability**. The report is visible
  only to the maintainers.

  > Status: this channel is **not yet enabled**. The repository owner turns
  > it on under *Settings → Code security → Private vulnerability
  > reporting → Enable*, or with
  > `gh api -X PUT repos/avinashamanchi/sepsis-vitals/private-vulnerability-reporting`.

- **Until it is enabled:** open a public issue titled *"Security contact
  request"* that contains no details. A maintainer will reply with a
  private way to send the report.

Please include:
- the affected component and commit;
- the steps to reproduce, using synthetic data only;
- the impact you expect.

We aim to acknowledge reports within 5 working days. This is a goal, not a
contractual service level.

## Supported versions

Only the latest commit on `main` receives fixes. There are no supported
releases for clinical or production use.

## Dependency security

| Check | Where | Gate |
|---|---|---|
| `pip-audit` on `requirements/deploy.txt` (what the API image installs) and `requirements/dev.txt` | CI `security` job (Python 3.11), and `test` jobs for each supported Python | Any known vulnerability fails CI |
| `npm audit` on the dashboard dependencies (all severities) | CI `frontend` job | Any known vulnerability fails CI |
| `bandit -ll` on `src/` | CI `security` job | Medium or high findings fail CI |
| Dependabot version updates: pip, npm, GitHub Actions, Docker base images | `.github/dependabot.yml`, weekly | Each PR runs the full CI |

### Remediation

1. Upgrade the affected package and regenerate the locks
   (`docs/dependencies.md`). Prefer the smallest version that fixes the
   issue, and check its changelog for breaking changes.
2. If no fix exists, or upgrading breaks compatibility (for example
   scikit-learn, which is pinned to the model artifact), add a **narrow,
   time-bound** exception:
   - Python: `--ignore-vuln <ID>` on the specific `pip-audit` call in
     `.github/workflows/ci.yml`. Add a comment giving the reason, why it is
     not exploitable here or what mitigates it, an owner, and a review date
     no more than 90 days away.
   - npm: an `overrides` entry, or a documented `npm audit` exception with
     the same fields.
   - Never use a broad exclusion such as `--audit-level=critical` or
     skipping a whole package.
3. Record the decision in the PR description so reviewers can check it.

No exceptions are in place today.

## Handling of secrets and data

- Secrets come from the environment or a secret store and are never
  committed. GitHub secret scanning and push protection are enabled on the
  repository.
- Personal data at rest is encrypted with keys from the PII keyring. To
  rotate those keys, or to respond to a suspected key exposure, see
  `docs/pii_key_rotation.md`. Rotation does not undo an exposure.
- CI uses throwaway secrets generated inside each job, runs on
  `pull_request` with read-only permissions, and exposes no repository
  secrets to pull requests.
