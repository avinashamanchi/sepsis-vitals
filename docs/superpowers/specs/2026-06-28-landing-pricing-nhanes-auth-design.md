# Landing Page, Pricing, NHANES Explorer & Firebase Auth — Design Spec

## Goal

Add a public-facing marketing layer (landing page + pricing) and an in-app NHANES population health explorer to the existing React clinical platform. Replace mock auth with Firebase Auth (Google sign-in + email/password).

## Architecture

The app gains 3 new React pages and a routing restructure. Public pages (landing, pricing) render without the sidebar/topbar layout. The NHANES explorer is an authenticated page inside the app. Firebase Auth replaces the current mock login flow, handling Google OAuth, email/password, password reset, and session persistence. No backend changes are required — Firebase handles auth client-side.

## Tech Stack

- React 18 + TypeScript + Tailwind CSS (existing)
- Firebase Auth SDK (`firebase` npm package)
- Recharts (existing — used for NHANES charts)
- react-router-dom (existing — add public routes)
- i18n via react-i18next (existing — add keys for new pages)

---

## 1. Routing & Page Structure

### Public routes (no AuthGuard, no sidebar/topbar)

| Route | Page | Purpose |
|-------|------|---------|
| `/` | Landing | Marketing hero, features, NHANES teaser, CTAs |
| `/pricing` | Pricing | 3 tiers, toggle, ROI calculator, feature matrix |
| `/login` | Login | Firebase Auth (Google + email/password) |

### Authenticated routes (sidebar/topbar layout, same as today)

| Route | Page | Change |
|-------|------|--------|
| `/dashboard` | Dashboard | Moved from `/`, no other changes |
| `/population` | Population | New — NHANES explorer |
| `/patients` | Patients | Unchanged |
| `/patients/:id` | PatientDetail | Unchanged |
| `/monitor` | Monitor | Unchanged |
| `/scores` | ScoreLab | Unchanged |
| `/predict` | Predict | Unchanged |
| `/analytics` | Analytics | Unchanged |
| `/alerts` | Alerts | Unchanged |
| `/admin` | Admin | Unchanged |

### Routing logic

- Unauthenticated user hits `/` — sees Landing page.
- Unauthenticated user hits any app route (`/dashboard`, `/patients`, etc.) — redirected to `/login`.
- Authenticated user hits `/` — redirected to `/dashboard`.
- Landing page "Get Started" / "Sign In" buttons go to `/login`.
- Pricing page "Start Free Trial" buttons go to `/login`.
- **Demo mode** (GitHub Pages, `isDemo` flag): skip auth checks, landing page still shows at `/` for unauthenticated visitors, but all app routes are accessible without login (same as current behavior).

---

## 2. Firebase Auth Integration

### Setup

- Add `firebase` npm package.
- `frontend/src/lib/firebase.ts` — Firebase app initialization. Config read from env vars:
  - `VITE_FIREBASE_API_KEY`
  - `VITE_FIREBASE_AUTH_DOMAIN`
  - `VITE_FIREBASE_PROJECT_ID`
  - `VITE_FIREBASE_STORAGE_BUCKET`
  - `VITE_FIREBASE_MESSAGING_SENDER_ID`
  - `VITE_FIREBASE_APP_ID`
- `frontend/src/lib/auth.ts` — Auth helper functions wrapping Firebase:
  - `signInWithGoogle()` — popup-based Google sign-in
  - `signInWithEmail(email, password)` — email/password sign-in
  - `signUpWithEmail(email, password)` — create account
  - `signOutUser()` — sign out
  - `resetPassword(email)` — send password reset email
  - `onAuthChange(callback)` — subscribe to auth state changes

### Connection to existing app

- On successful Firebase sign-in, call `user.getIdToken()` to get the Firebase ID token.
- Store the token and user info (`email`, `displayName`, `photoURL`) in Zustand via `useStore.setAuth()`.
- Sign-out calls `signOutUser()` from `auth.ts` + `useStore.logout()`.
- On app load, `onAuthStateChanged` listener in `App.tsx` checks if user is already signed in. If so, auto-populate Zustand store — user stays logged in across refreshes (Firebase persists auth state in IndexedDB).
- If Firebase token expires, Firebase auto-refreshes it.

### Login page updates

- "Sign in with Google" button (prominent, top of form).
- Divider ("or sign in with email").
- Email + password fields.
- "Forgot password?" link — triggers `sendPasswordResetEmail()`.
- "Create account" toggle — shows sign-up form with email + password + confirm password.
- Error states: invalid credentials, email already in use, weak password (Firebase provides error codes).
- On success, redirect to `/dashboard`.

---

## 3. Landing Page

Public page at `/`. Full-width, no sidebar/topbar. Dark theme matching the app.

### Sections

**Hero:**
- Headline: "Detect Sepsis Hours Before It's Too Late"
- Subheadline about AI-powered early warning for hospitals.
- "Get Started" (primary CTA → `/login`) and "View Pricing" (secondary → `/pricing`) buttons.
- Stats bar: "6hr earlier detection", "40% mortality reduction", "99% specificity".
- Subtle green radial glow background.

**Problem:**
- 3 cards with red accent border:
  - 270,000 deaths/year in the US from sepsis
  - $27B annual cost to healthcare system
  - Every hour of delayed treatment increases mortality 4-8%

**Features:**
- 6 feature cards with hover effects and tag badges:
  - ML Prediction (AI tag)
  - Real-time Monitoring (Clinical tag)
  - NHANES Benchmarking (Data tag)
  - Clinical Scoring — qSOFA, NEWS2, SOFA (Clinical tag)
  - FHIR Integration (Integration tag)
  - Multi-language Support (Platform tag)

**NHANES teaser:**
- "Powered by Population Health Data" header.
- Brief visual — sample percentile chart using Recharts.
- "Built on CDC NHANES data covering 11 survey cycles (1999-2023)".
- CTA to sign up to explore full data.

**Compliance/social proof:**
- HIPAA, SOC 2, HL7 FHIR badge row.

**Footer:**
- Links to pricing, login.
- Copyright.

---

## 4. Pricing Page

Public page at `/pricing`. Same full-width marketing layout as landing.

### Header

- "Simple, Transparent Pricing" headline.
- "Per-bed pricing that scales with your hospital" subtext.

### Annual/Monthly toggle

- Toggle switch. Annual shows ~17% savings.
- "Save 17%" badge appears when annual is selected.

### 3 pricing cards

| | Starter | Professional (featured) | Enterprise |
|---|---|---|---|
| Monthly | $15/bed/mo | $32/bed/mo | $50/bed/mo |
| Annual | $150/bed/yr | $320/bed/yr | $500/bed/yr |
| Scoring | qSOFA, NEWS2 | + ML predictions | + custom models |
| Alerts | Email | + real-time WebSocket | + escalation chains |
| Monitoring | Basic dashboard | + NHANES benchmarking | + FHIR integration |
| Support | Community | Priority | Dedicated + SLA |
| CTA | "Get Started" → `/login` | "Start Free Trial" → `/login` | "Contact Sales" (mailto or form) |

Professional card gets a green border highlight and "Most Popular" badge.

### Feature comparison matrix

Full-width table below the cards with checkmarks/dashes for all features across tiers.

### ROI Calculator

- Inputs: number of beds (slider or input), average sepsis cases per year (slider or input).
- Outputs: estimated annual cost savings, estimated lives saved, ROI percentage.
- Based on published sepsis economics: average sepsis case costs ~$35,000; early detection reduces cost by ~25%; reduces mortality by ~20%.
- Formula:
  - `savings = sepsisCases * 35000 * 0.25`
  - `livesSaved = sepsisCases * 0.20`
  - `cost = beds * pricePerBed * 12`
  - `roi = ((savings - cost) / cost) * 100`

---

## 5. NHANES Population Health Explorer

Authenticated route at `/population`. Renders inside the sidebar/topbar layout.

### Sidebar nav

Add entry to `NAV_ITEMS` in `Sidebar.tsx`:
- Icon: `Globe` from lucide-react
- Label: "Population" (i18n key: `nav.population`)
- Route: `/population`
- Position: after Analytics, before Alerts

### Data module

Port the old 585-line `docs/data/nhanes.js` to `frontend/src/data/nhanes.ts`:
- TypeScript interfaces for all data structures.
- Covers: blood pressure (systolic/diastolic), heart rate, respiratory rate, SpO2, CBC, CRP.
- Stratified by: age group (18-29 through 80+), sex (Male/Female), ethnicity (6 groups).
- 11 survey cycles (1999-2023).
- Percentile data: p5, p25, p50, p75, p95 plus mean and SD.
- Prevalence data: hypertension by age, ethnicity comparisons.
- Trend data: population means across survey cycles.
- Utility functions: `getDistribution(vital, sex, ageGroup)`, `getTrend(vital)`, `getByEthnicity(vital)`.

### Page layout

**Filter bar** (sticky below topbar):
- Age group dropdown (18-29, 30-39, 40-49, 50-59, 60-69, 70-79, 80+)
- Sex toggle (Male / Female)
- Ethnicity dropdown (All, Non-Hispanic White, Non-Hispanic Black, Mexican-American, Other Hispanic, Asian, Other/Multi)
- Vital sign selector (Blood Pressure, Heart Rate, Respiratory Rate, SpO2)

**Distribution section:**
- Bar chart (Recharts `BarChart`) showing percentile distribution (p5, p25, p50, p75, p95) for the selected vital + demographic.
- Color-coded bars (green for normal range, amber for borderline, red for abnormal).

**Trends section:**
- Line chart (Recharts `LineChart`) showing population mean across survey cycles (1999-2023).
- For BP: both systolic and diastolic lines.

**Reference ranges table:**
- Table with columns: Age Group, Mean, SD, p5, p25, p50, p75, p95.
- Row for each age group. Selected demographic's row highlighted.

**Prevalence section:**
- Bar chart showing hypertension prevalence by age group.
- Grouped bar chart showing ethnicity comparison for selected vital.

---

## 6. File Structure

### New files

| File | Purpose |
|------|---------|
| `frontend/src/lib/firebase.ts` | Firebase app initialization |
| `frontend/src/lib/auth.ts` | Auth helper functions |
| `frontend/src/data/nhanes.ts` | NHANES population data (TypeScript) |
| `frontend/src/pages/Landing.tsx` | Public landing page |
| `frontend/src/pages/Pricing.tsx` | Public pricing page |
| `frontend/src/pages/Population.tsx` | NHANES explorer (authenticated) |

### Modified files

| File | Change |
|------|--------|
| `frontend/src/App.tsx` | Add public routes, move dashboard to `/dashboard`, redirect logic, Firebase auth listener |
| `frontend/src/pages/Login.tsx` | Replace mock auth with Firebase (Google + email/password + reset) |
| `frontend/src/components/Sidebar.tsx` | Add Population nav item |
| `frontend/src/components/BottomNav.tsx` | Add Population to mobile nav |
| `frontend/src/stores/useStore.ts` | Wire Firebase auth state, add photoURL to user type |
| `frontend/package.json` | Add `firebase` dependency |

### Unchanged

- Dashboard, Patients, PatientDetail, Monitor, Predict, Analytics, Alerts, Admin, ScoreLab — all untouched.
- Backend — no changes. Firebase handles auth client-side.
- Existing i18n structure — add new keys for new pages, don't modify existing keys.
