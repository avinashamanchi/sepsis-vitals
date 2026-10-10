#!/usr/bin/env bash
# Guard for the published (GitHub Pages) demo site. Usage: check_published_site.sh <dist-dir>
set -euo pipefail
dist="${1:?usage: check_published_site.sh <dist-dir>}"
fail() { echo "publish guard: $*" >&2; exit 1; }

test -f "$dist/index.html" || fail "no index.html in $dist"
# No key material, tokens or environment files.
if grep -rIlE "BEGIN [A-Z ]*PRIVATE KEY|sk-ant-[A-Za-z0-9_-]{8,}|gh[opsu]_[A-Za-z0-9]{20,}|SEPSIS_(PII|JWT)_[A-Z]*=" "$dist"; then
  fail "secret-like content found"
fi
if find "$dist" \( -name "*.map" -o -name ".env*" \) | grep -q .; then
  fail "source maps or env files would be published"
fi
# Claims retired in the validation-first rewrite must not come back.
if grep -rIliE "99% specificity|hours earlier than|reduces? (sepsis )?mortality by" "$dist"; then
  fail "unsupported performance claim found"
fi
# The research-use disclaimer and synthetic-data labelling must ship.
grep -rIlq "Not for patient care" "$dist" || fail "research-use disclaimer missing"
grep -rIliq "synthetic" "$dist" || fail "synthetic-data labelling missing"
echo "publish guard: ok ($dist)"
