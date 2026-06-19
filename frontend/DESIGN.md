# Sepsis Vitals Design System

## Philosophy
Clinical-grade dark UI designed for ICU environments. Minimal, data-dense, zero-distraction. Every pixel serves patient safety.

## Colors

### Backgrounds
- `--bg-void`: `#04080f` — primary canvas
- `--bg-surface`: `#0a1120` — card/panel backgrounds
- `--bg-elevated`: `#111d2e` — hover states, active items
- `--bg-overlay`: `#182438` — modals, dropdowns

### Accents
- `--accent`: `#00ff9d` — primary action, success, healthy
- `--accent-dim`: `#00cc7d` — hover state
- `--warning`: `#ffb830` — moderate risk, caution
- `--danger`: `#ff3b5c` — high/critical risk, alerts
- `--info`: `#38b4ff` — informational, links

### Text
- `--text-primary`: `#e8f4ff` — headings, primary content
- `--text-secondary`: `#8ba8cc` — labels, descriptions
- `--text-muted`: `#4a6080` — disabled, timestamps

### Risk Levels (clinical)
- `--risk-low`: `#00ff9d`
- `--risk-moderate`: `#ffb830`
- `--risk-high`: `#ff6b35`
- `--risk-critical`: `#ff3b5c`

## Typography
- **Headings**: `"Syne", sans-serif` — weights 600, 700, 800
- **Body/Mono**: `"JetBrains Mono", monospace` — weights 300, 400, 500
- **Scale**: 12px / 14px / 16px / 20px / 24px / 32px / 48px

## Spacing
- Base unit: 4px
- Scale: 4 / 8 / 12 / 16 / 24 / 32 / 48 / 64 / 96

## Border Radius
- `--radius-sm`: 4px — buttons, inputs
- `--radius-md`: 8px — cards, panels
- `--radius-lg`: 12px — modals, large containers
- `--radius-full`: 9999px — pills, avatars

## Shadows
- `--shadow-sm`: `0 1px 2px rgba(0,0,0,0.3)`
- `--shadow-md`: `0 4px 12px rgba(0,0,0,0.4)`
- `--shadow-lg`: `0 8px 24px rgba(0,0,0,0.5)`
- `--shadow-glow`: `0 0 20px rgba(0,255,157,0.15)` — accent glow for alerts

## Component Patterns

### Cards
- Background: `--bg-surface`
- Border: 1px `rgba(255,255,255,0.06)`
- Colored top bar (4px) for category coding

### Alert Items
- Left border (3px) color-coded by risk level
- Timestamp in muted text, right-aligned
- Action buttons on hover only

### Stat Tiles
- Large number in accent/risk color
- Label below in secondary text
- Subtle background gradient

### Charts
- Dark grid lines at 0.06 opacity
- Accent-colored data series
- No chart backgrounds (transparent)

## Responsive Breakpoints
- Mobile: < 768px (bottom nav, stacked cards)
- Tablet: 768px–1024px (sidebar + content)
- Desktop: > 1024px (full layout)

## Animations
- Transitions: 150ms ease-out (interactions), 300ms ease (layout)
- Alert pulse: `@keyframes pulse { 0%, 100% { opacity: 1 } 50% { opacity: 0.6 } }` — 2s for critical
- No decorative animations. Motion only for state communication.
