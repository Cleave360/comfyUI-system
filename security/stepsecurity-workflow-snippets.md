# Kindred Systems AI — StepSecurity Workflow Snippets
## Drop-in security monitoring for GitHub Actions
**Classification:** Internal Use Only
**Version:** 1.0 | March 2026

---

> ⚠️ **Before You Start**
> - **Community Tier:** Public repos only. Free, 10,000 runs/week.
> - **Enterprise Tier:** Required for private repos. 14-day free trial available.
> - Install the StepSecurity GitHub App first: https://github.com/apps/stepsecurity-actions-security
> - View your insights dashboard at: https://app.stepsecurity.io

---

## Phase 1 — Audit Mode (Start Here)

Add this as the **very first step** in every job across all repos.
It observes and reports — does NOT block anything yet.

```yaml
steps:
  - name: Harden Runner
    uses: step-security/harden-runner@v2
    with:
      egress-policy: audit   # Observe only — safe to add immediately
      disable-sudo: false    # Set true if your job doesn't need sudo
```

After your workflow runs, look in the Actions log for a link like:
```
View security insights: https://app.stepsecurity.io/github/...
```
Click it to see every outbound network call and file write your pipeline made.

---

## Phase 2 — Block Mode (After Review)

Once you've reviewed the audit output and know what's legitimate,
switch to block mode and whitelist only the domains your repo actually needs.

```yaml
steps:
  - name: Harden Runner
    uses: step-security/harden-runner@v2
    with:
      egress-policy: block
      allowed-endpoints: >
        github.com:443
        api.github.com:443
        objects.githubusercontent.com:443
        pypi.org:443
        files.pythonhosted.org:443
```

> Replace the `allowed-endpoints` list with what you observed in audit mode.
> Any unlisted outbound call will be blocked and flagged.

---

## Full Workflow Templates by Repo Type

### 🐍 Python Repo

```yaml
# .github/workflows/security-audit.yml
name: Security Audit

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  audit:
    runs-on: ubuntu-latest
    permissions:
      contents: read

    steps:
      - name: Harden Runner
        uses: step-security/harden-runner@v2
        with:
          egress-policy: audit

      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run security scan
        run: |
          pip install pip-audit
          pip-audit --requirement requirements.txt
```

---

### 🟨 Node.js / JavaScript Repo

```yaml
# .github/workflows/security-audit.yml
name: Security Audit

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  audit:
    runs-on: ubuntu-latest
    permissions:
      contents: read

    steps:
      - name: Harden Runner
        uses: step-security/harden-runner@v2
        with:
          egress-policy: audit

      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Node
        uses: actions/setup-node@v4
        with:
          node-version: '20'

      - name: Install dependencies
        run: npm ci

      - name: Run npm audit
        run: npm audit --audit-level=moderate
```

---

### 🔴 Redis / Infrastructure Repo

```yaml
# .github/workflows/security-audit.yml
name: Security Audit

on:
  push:
    branches: [ main ]
  schedule:
    - cron: '0 6 * * 1'   # Every Monday at 6am — weekly automated scan

jobs:
  audit:
    runs-on: ubuntu-latest
    permissions:
      contents: read

    steps:
      - name: Harden Runner
        uses: step-security/harden-runner@v2
        with:
          egress-policy: audit

      - name: Checkout
        uses: actions/checkout@v4

      - name: Scan for hardcoded secrets
        uses: trufflesecurity/trufflehog@main
        with:
          path: ./
          base: main
          head: HEAD

      - name: Check for exposed config files
        run: |
          echo "Checking for .env files committed to repo..."
          find . -name ".env" -not -path "./.git/*" && echo "WARNING: .env file found" || echo "OK: No .env files found"
          find . -name "*.pem" -not -path "./.git/*" && echo "WARNING: PEM file found" || echo "OK: No PEM files found"
          find . -name "*.key" -not -path "./.git/*" && echo "WARNING: Key file found" || echo "OK: No key files found"
```

---

### 🌐 General Purpose (Unknown Stack)

Use this when you're not sure what the repo uses — it covers the basics for any type:

```yaml
# .github/workflows/security-audit.yml
name: Security Audit

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  audit:
    runs-on: ubuntu-latest
    permissions:
      contents: read

    steps:
      - name: Harden Runner
        uses: step-security/harden-runner@v2
        with:
          egress-policy: audit

      - name: Checkout
        uses: actions/checkout@v4

      - name: Scan for secrets
        uses: trufflesecurity/trufflehog@main
        with:
          path: ./
          base: main
          head: HEAD

      - name: Check dependency files present
        run: |
          echo "=== Dependency files found ==="
          find . -name "requirements.txt" -o -name "package.json" \
                 -o -name "Pipfile" -o -name "go.mod" \
                 -o -name "Gemfile" -o -name "composer.json" \
            | grep -v node_modules | grep -v .git
```

---

## Repo Audit Checklist

Use this as you work through each Kindred Systems repo:

```
Repo Name: ______________________
Date Audited: ___________________
Visibility: [ ] Public  [ ] Private
Workflow file added: [ ] Yes  [ ] No

StepSecurity Results:
[ ] Audit mode run completed
[ ] Network events reviewed
[ ] File write events reviewed
[ ] Unexpected outbound calls found: _______________
[ ] Secrets found in repo: [ ] Yes (URGENT) [ ] No
[ ] .env or key files committed: [ ] Yes (URGENT) [ ] No
[ ] Dependencies scanned: [ ] Yes [ ] No
[ ] Known CVEs in dependencies: [ ] Yes [ ] No

Next Action: ____________________
Priority: [ ] Critical [ ] High [ ] Medium [ ] Low
```

---

## Quick Reference — Egress Policy Options

| Policy | Behaviour | When to Use |
|--------|-----------|-------------|
| `audit` | Log all outbound calls, block nothing | First pass on every repo |
| `block` | Block all calls not in allowlist | After audit review |
| `block` + `allowed-endpoints` | Block everything except listed domains | Production hardening |

---

## Common Allowed Endpoints by Stack

Copy the relevant block into your `allowed-endpoints` after audit review:

**Python / PyPI**
```
pypi.org:443
files.pythonhosted.org:443
```

**Node / npm**
```
registry.npmjs.org:443
```

**GitHub core (always include)**
```
github.com:443
api.github.com:443
objects.githubusercontent.com:443
```

**Redis / General networking tools**
```
# Add specific endpoints observed in your audit
```

---

*Kindred Systems AI — Internal Security Document*
*Do not distribute outside the organisation*
