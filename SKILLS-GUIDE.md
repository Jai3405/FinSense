# Claude Code Skills — Reference Guide
> Your complete cheat sheet for when to use what

---

## How to invoke a skill
Type the skill name with a `/` prefix before your request:
```
/frontend-design build me the FinScore page
/web-design-guidelines review the dashboard
```
Superpowers skills (marked ⚡) run automatically — no typing needed.

---

## 🎨 Design & Frontend

### `/frontend-design`
**When:** Any time you're building or redesigning a UI component, page, or screen.
**Use it for:**
- Building new dashboard pages (FinScore, Portfolio, Discover, etc.)
- Redesigning existing pages that look too generic
- Building landing pages, marketing pages, sign-in screens
- Any time you want the output to look premium, not AI-generated

**Example:**
```
/frontend-design build the FinScore page with a 0-100 score display for a stock
```

---

### `/web-design-guidelines`
**When:** After building something — run a review before you ship.
**Use it for:**
- Checking accessibility (contrast ratios, ARIA labels, keyboard nav)
- Catching bad UX patterns (confusing flows, unclear CTAs)
- Making sure the UI follows best practices
- Pre-launch audit of any page

**Example:**
```
/web-design-guidelines review the sidebar and dashboard home page
```

---

### `/theme-factory`
**When:** You want to establish or change the visual theme across a surface.
**Use it for:**
- Formalizing SPIKE's design tokens into a Tailwind config
- Generating a consistent color/typography/spacing system
- Creating themed variants (dark mode, high contrast)
- Styling reports, slides, or documents

**Example:**
```
/theme-factory create a proper design token system from the SPIKE brand brief
```

---

## ⚡ Performance & Code Quality

### `/vercel-react-best-practices`
**When:** Before shipping any React/Next.js feature, or during a code review.
**Use it for:**
- Eliminating request waterfalls (the #1 performance killer)
- Fixing unnecessary re-renders
- Reducing bundle size (barrel imports, dynamic imports)
- Optimizing data fetching patterns
- Any time the app feels slow

**Example:**
```
/vercel-react-best-practices review the dashboard layout for performance issues
```

---

## 🎬 Creative & Generative

### `/algorithmic-art`
**When:** You want data-driven, generative, or animated visual elements.
**Use it for:**
- FinScore visual — a unique animated score display
- Portfolio heatmaps and data visualizations
- Generative background patterns for the dashboard
- Any place where a static chart feels boring

**Example:**
```
/algorithmic-art create a generative visualization for a stock's volatility pattern
```

---

### `/remotion-best-practices`
**When:** Building any video or animation component with Remotion.
**Use it for:**
- Monthly portfolio performance summary videos
- Animated onboarding sequences
- Stock movement replays
- Any programmatic video content

**Example:**
```
/remotion-best-practices build a 30-second portfolio performance recap video component
```

---

## ⚡ Superpowers (Automatic — no typing needed)

These run in the background automatically. Listed here so you know what's happening.

| Skill | Triggers when... |
|-------|-----------------|
| `brainstorming` | Before ANY new feature or creative work — explores requirements first |
| `writing-plans` | Big multi-step feature — writes a full plan before touching code |
| `executing-plans` | Executing a written plan with checkpoints |
| `systematic-debugging` | Something is broken — structured root cause analysis |
| `test-driven-development` | Implementing a feature — writes tests first |
| `dispatching-parallel-agents` | 2+ independent tasks — runs them simultaneously |
| `requesting-code-review` | After completing a major feature |
| `verification-before-completion` | Before claiming anything is "done" — actually verifies it works |
| `finishing-a-development-branch` | When implementation is complete, guides merge/PR decision |
| `subagent-driven-development` | Executing plans with independent parallel tasks |
| `using-git-worktrees` | Isolated feature work that needs its own workspace |

---

## 🧠 Context7 MCP (Automatic)

Not a slash command — runs automatically whenever I need live documentation.

**What it does:** Pulls the latest official docs for any library in real-time instead of guessing from training data.

**Libraries it helps with for SPIKE:**
- Next.js 15 (App Router, Server Components, streaming)
- React 19 (hooks, concurrent features)
- Tailwind CSS v4
- FastAPI (Python backend)
- PyTorch / scikit-learn (ML models)
- Any other library we add

---

## 📋 Quick Decision Tree

```
Building new UI?          → /frontend-design
Reviewing existing UI?    → /web-design-guidelines
Slow or buggy React code? → /vercel-react-best-practices
Want design system?       → /theme-factory
Data visualization?       → /algorithmic-art
Video content?            → /remotion-best-practices
Everything else?          → Just talk normally, superpowers handle it
```

---

## 🚀 Power Combos for SPIKE

**Building a new page end-to-end:**
```
1. /frontend-design build the [page name]
2. /web-design-guidelines review what we just built
3. /vercel-react-best-practices optimize it
```

**New data visualization:**
```
1. /algorithmic-art create a [visualization type] for [data]
2. /vercel-react-best-practices check it for performance
```

**Design system work:**
```
1. /theme-factory generate SPIKE's full design token system
2. /web-design-guidelines audit against accessibility standards
```

---

*Last updated: February 2026*
