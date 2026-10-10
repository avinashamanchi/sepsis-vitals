# Dependencies

## Installation sets

| Set | Install | What it is for |
|---|---|---|
| base | `pip install .` | Rule-based scores (`sepsis_vitals.scores`), data helpers |
| `api` | `.[api]` | HTTP API, authentication, database, Alembic, Redis-backed revocation. Without `[ml]` the API runs, and `/model/status` reports `unavailable` |
| `ml` | `.[ml]` | Model inference: scikit-learn (pinned to the version the artifact was saved with), joblib, and the LightGBM/XGBoost runtimes the trainer may select |
| `train` | `.[ml,train]` | Training and explanation reports (`shap`). Always install it together with `[ml]` |
| `copilot` | `.[copilot]` | Anthropic SDK for the frozen copilot feature |
| `integrations` | `.[integrations]` | Stripe, Twilio, Africa's Talking, Web Push (frozen billing and SMS/push channels) |
| `dev` | `.[dev]` | Tests, linters, type checking, `pip-audit` |

The API container installs `[api,ml]` only. Optional SDKs are imported
lazily inside the features that use them, so a set never needs another
set's packages. CI (`extras` job) installs each set alone in a clean
environment on Python 3.10 and 3.12, runs `pip check`, and runs
`scripts/smoke_extras.py`. That script also asserts that the other sets'
packages are absent.

## Locks

| File | Covers | Used by |
|---|---|---|
| `requirements/deploy.txt` | base + `[api,ml]` | `docker/Dockerfile` (`pip install --require-hashes`) |
| `requirements/dev.txt` | base + every extra + `dev` | CI test, lint, typecheck, security and Postgres jobs |

Both are universal locks for Python 3.10–3.12. Where a package's newest
release drops an older Python, environment markers select a version per
Python (for example numpy and pandas). Every pin carries hashes, and pip
refuses unhashed or unpinned packages.

The project itself is installed afterwards with `pip install --no-deps .`
(or `-e .`). Its build backend (setuptools) is fetched by pip's build
isolation and is not hash-locked; this is a known limitation.

`scripts/check_lock.py` (CI lint job) checks that every direct requirement
in `pyproject.toml` is pinned exactly once per supported Python and
satisfies its specifier. A `pyproject.toml` edit without a lock update
fails CI.

## Updating

```bash
uv pip compile pyproject.toml --extra api --extra ml \
  --universal --python-version 3.10 --generate-hashes --no-header -o requirements/deploy.txt
```

```bash
uv pip compile pyproject.toml --extra api --extra ml --extra train --extra copilot \
  --extra integrations --extra dev \
  --universal --python-version 3.10 --generate-hashes --no-header -o requirements/dev.txt
```

```bash
python scripts/check_lock.py
```

Then let CI run the full matrix, Docker, compose-smoke and the Postgres
jobs on the change. To move a single package, add `--upgrade-package NAME`.
Review the diff for major-version jumps before merging. `scikit-learn`
stays pinned to the version in `models/manifest.json` until the model is
retrained and its manifest rebuilt.

Dependabot proposes updates weekly (`.github/dependabot.yml`). Because the
pins live in the lock files, regenerate the locks as above when acting on
a Dependabot PR for `pyproject.toml`.

## Platforms

| Platform | Status |
|---|---|
| Linux x86_64, Python 3.10/3.11/3.12 | Tested in CI (tests, extras, Docker image on 3.11) |
| macOS arm64, Python 3.11 | Tested locally for every set. LightGBM and XGBoost need the OpenMP runtime (`brew install libomp`); without it the trainer skips them, and the smoke test reports this as a platform limitation |
| Windows | Not tested |
