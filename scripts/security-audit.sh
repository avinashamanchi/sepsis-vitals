#!/usr/bin/env bash
# security-audit.sh — Run dependency vulnerability scans for CI/CD.
#
# Usage: ./scripts/security-audit.sh [--fix]
#
# Exits non-zero if vulnerabilities are found (suitable for CI gates).

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

EXIT_CODE=0

echo "=== Sepsis-Vitals Security Audit ==="
echo ""

# ── Python dependency audit ──────────────────────────────────────────────────

echo -e "${YELLOW}[1/3] Python dependency audit (pip-audit)${NC}"
if command -v pip-audit &>/dev/null; then
    if pip-audit --strict --desc 2>&1; then
        echo -e "${GREEN}  ✓ No known Python vulnerabilities${NC}"
    else
        echo -e "${RED}  ✗ Python vulnerabilities found${NC}"
        EXIT_CODE=1
    fi
else
    echo -e "${YELLOW}  ⚠ pip-audit not installed. Install with: pip install pip-audit${NC}"
    echo "  Falling back to pip check..."
    if pip check 2>&1; then
        echo -e "${GREEN}  ✓ pip check passed (no broken deps)${NC}"
    else
        echo -e "${RED}  ✗ pip check failed${NC}"
        EXIT_CODE=1
    fi
fi

echo ""

# ── Node.js dependency audit ────────────────────────────────────────────────

echo -e "${YELLOW}[2/3] Node.js dependency audit (npm audit)${NC}"
if [ -d "frontend" ] && [ -f "frontend/package-lock.json" ]; then
    cd frontend
    AUDIT_OUTPUT=$(npm audit --omit=dev 2>&1) || true
    if echo "$AUDIT_OUTPUT" | grep -q "found 0 vulnerabilities"; then
        echo -e "${GREEN}  ✓ No known npm vulnerabilities${NC}"
    else
        echo "$AUDIT_OUTPUT" | tail -20
        # Only fail on high/critical
        if echo "$AUDIT_OUTPUT" | grep -qE "(high|critical)"; then
            echo -e "${RED}  ✗ High/critical npm vulnerabilities found${NC}"
            EXIT_CODE=1
        else
            echo -e "${YELLOW}  ⚠ Low/moderate npm vulnerabilities (non-blocking)${NC}"
        fi
    fi
    cd ..
else
    echo "  Skipped (no frontend/package-lock.json)"
fi

echo ""

# ── Secret scanning ─────────────────────────────────────────────────────────

echo -e "${YELLOW}[3/3] Secret scanning (git-tracked files)${NC}"
SECRETS_FOUND=0

# Check for hardcoded secrets in tracked files
PATTERNS=(
    'AKIA[0-9A-Z]{16}'           # AWS access key
    'sk-[a-zA-Z0-9]{20,}'        # Stripe/OpenAI secret key
    'ghp_[a-zA-Z0-9]{36}'        # GitHub personal access token
    'password\s*=\s*["\x27][^"\x27]{8,}'  # Hardcoded passwords
)

for pattern in "${PATTERNS[@]}"; do
    if git grep -lP "$pattern" -- ':!scripts/security-audit.sh' ':!*.md' 2>/dev/null; then
        echo -e "${RED}  ✗ Possible hardcoded secret matching: $pattern${NC}"
        SECRETS_FOUND=1
    fi
done

# Check for committed .env files
if git ls-files | grep -qE '\.env$|\.env\.' 2>/dev/null; then
    echo -e "${RED}  ✗ .env file(s) committed to git${NC}"
    SECRETS_FOUND=1
fi

if [ "$SECRETS_FOUND" -eq 0 ]; then
    echo -e "${GREEN}  ✓ No obvious secrets in tracked files${NC}"
else
    EXIT_CODE=1
fi

echo ""
echo "=== Audit complete ==="

if [ "$EXIT_CODE" -eq 0 ]; then
    echo -e "${GREEN}All checks passed.${NC}"
else
    echo -e "${RED}Some checks failed. Review output above.${NC}"
fi

exit $EXIT_CODE
