"use client";

import { motion, useInView } from "framer-motion";
import { useRef, useEffect, useState } from "react";
import {
  Brain,
  Shield,
  Layers,
  Bot,
  BookOpen,
  Bell,
  BarChart3,
  Compass,
  MessageCircle,
  Gauge,
  ArrowUpRight,
  TrendingUp,
  TrendingDown,
} from "lucide-react";
import { Reveal } from "./reveal";

/* ─────────────────────────────────────────────
   MICRO EXPERIENCES
   Each is self-contained, designed to fill space
   ───────────────────────────────────────────── */

function FinScoreGauge() {
  const [score, setScore] = useState(0);
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: true });

  useEffect(() => {
    if (!isInView) return;
    let current = 0;
    const target = 82;
    const timer = setInterval(() => {
      current += 1;
      setScore(current);
      if (current >= target) clearInterval(timer);
    }, 18);
    return () => clearInterval(timer);
  }, [isInView]);

  const circumference = 2 * Math.PI * 58;
  const strokeDashoffset = circumference - (score / 100) * circumference;
  const color =
    score >= 70
      ? "#22C55E"
      : score >= 40
        ? "#C9A96E"
        : "#EF4444";

  return (
    <div ref={ref} className="flex items-center justify-center h-full">
      <div className="relative">
        <svg width="160" height="160" viewBox="0 0 160 160">
          <circle
            cx="80"
            cy="80"
            r="58"
            fill="none"
            stroke="rgba(255,255,255,0.04)"
            strokeWidth="10"
          />
          <motion.circle
            cx="80"
            cy="80"
            r="58"
            fill="none"
            stroke={color}
            strokeWidth="10"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            transform="rotate(-90 80 80)"
            style={{ filter: `drop-shadow(0 0 12px ${color}50)` }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-4xl font-bold text-spike-cream tabular-nums">
            {score}
          </span>
          <span className="text-[11px] text-spike-muted mt-1">HDFCBANK</span>
        </div>
      </div>
    </div>
  );
}

function LiveTickerStrip() {
  const tickers = [
    { symbol: "HDFCBANK", price: "₹1,642", change: "+2.1%", up: true },
    { symbol: "TCS", price: "₹3,845", change: "-0.4%", up: false },
    { symbol: "RELIANCE", price: "₹2,478", change: "+1.8%", up: true },
    { symbol: "INFY", price: "₹1,521", change: "+0.7%", up: true },
    { symbol: "TATAMOTORS", price: "₹642", change: "-1.2%", up: false },
    { symbol: "WIPRO", price: "₹485", change: "+0.3%", up: true },
    { symbol: "ICICIBANK", price: "₹1,089", change: "+1.5%", up: true },
  ];

  return (
    <div className="h-full flex flex-col justify-center space-y-2">
      {tickers.map((t, i) => (
        <motion.div
          key={t.symbol}
          className="flex items-center justify-between px-4 py-2.5 rounded-xl bg-white/[0.02] hover:bg-white/[0.05] transition-colors"
          initial={{ opacity: 0, x: -30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: i * 0.06, duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
        >
          <div className="flex items-center gap-3">
            <div
              className={`w-1.5 h-1.5 rounded-full ${t.up ? "bg-spike-bull" : "bg-spike-bear"}`}
            />
            <span className="text-sm font-medium text-spike-cream/90">
              {t.symbol}
            </span>
          </div>
          <div className="flex items-center gap-4">
            <span className="text-sm text-spike-muted tabular-nums">
              {t.price}
            </span>
            <span
              className={`text-sm font-medium tabular-nums w-14 text-right ${t.up ? "text-spike-bull" : "text-spike-bear"}`}
            >
              {t.change}
            </span>
          </div>
        </motion.div>
      ))}
    </div>
  );
}

function StrategyPreview() {
  const metrics = [
    { label: "CAGR", value: "24.3%" },
    { label: "Sharpe", value: "1.82" },
    { label: "Max DD", value: "-12.1%" },
    { label: "Win Rate", value: "68%" },
  ];

  return (
    <div className="h-full flex flex-col justify-between">
      <div>
        <div className="flex items-center gap-2 mb-6">
          <div className="h-6 px-2.5 rounded-full bg-spike-sage/15 flex items-center">
            <span className="text-[11px] font-medium text-spike-sage">
              Backtested · 3yr
            </span>
          </div>
          <span className="text-sm text-spike-muted">
            &ldquo;Quality momentum mid-caps&rdquo;
          </span>
        </div>

        {/* Equity curve */}
        <svg
          viewBox="0 0 400 120"
          className="w-full h-28"
          preserveAspectRatio="none"
        >
          <defs>
            <linearGradient id="eq-fill" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="rgba(99, 140, 130, 0.25)" />
              <stop offset="100%" stopColor="rgba(99, 140, 130, 0)" />
            </linearGradient>
          </defs>
          <motion.path
            d="M0,100 Q30,95 60,88 T120,70 T180,55 T240,42 T300,28 T360,18 T400,10"
            fill="none"
            stroke="#638C82"
            strokeWidth="2.5"
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 2.5, ease: "easeOut" }}
          />
          <motion.path
            d="M0,100 Q30,95 60,88 T120,70 T180,55 T240,42 T300,28 T360,18 T400,10 V120 H0 Z"
            fill="url(#eq-fill)"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1.8, duration: 1 }}
          />
        </svg>
      </div>

      <div className="grid grid-cols-4 gap-3 mt-4">
        {metrics.map((m, i) => (
          <motion.div
            key={m.label}
            className="text-center p-2.5 rounded-xl bg-white/[0.03]"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.1 + 0.5 }}
          >
            <div className="text-base font-semibold text-spike-cream tabular-nums">
              {m.value}
            </div>
            <div className="text-[10px] text-spike-muted mt-0.5">{m.label}</div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}

function RiskHeatmap() {
  const sectors = [
    { name: "IT", weight: 64, risk: "high" as const },
    { name: "Banking", weight: 48, risk: "low" as const },
    { name: "Pharma", weight: 36, risk: "med" as const },
    { name: "Auto", weight: 28, risk: "med" as const },
    { name: "Energy", weight: 24, risk: "low" as const },
  ];

  const riskStyles = {
    low: { bar: "bg-spike-bull/60", dot: "bg-spike-bull" },
    med: { bar: "bg-spike-gold/60", dot: "bg-spike-gold" },
    high: { bar: "bg-spike-bear/60", dot: "bg-spike-bear" },
  };

  return (
    <div className="h-full flex flex-col justify-center space-y-4">
      {sectors.map((s, i) => (
        <motion.div
          key={s.name}
          className="flex items-center gap-4"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: i * 0.08, duration: 0.5 }}
        >
          <span className="text-xs font-medium text-spike-muted w-14 text-right">
            {s.name}
          </span>
          <div className="flex-1 h-2.5 rounded-full bg-white/[0.04] overflow-hidden">
            <motion.div
              className={`h-full rounded-full ${riskStyles[s.risk].bar}`}
              initial={{ width: "0%" }}
              animate={{ width: `${s.weight}%` }}
              transition={{
                delay: i * 0.1 + 0.2,
                duration: 0.8,
                ease: [0.16, 1, 0.3, 1],
              }}
            />
          </div>
          <div className="flex items-center gap-2 w-16">
            <div
              className={`w-1.5 h-1.5 rounded-full ${riskStyles[s.risk].dot}`}
            />
            <span className="text-xs text-spike-muted tabular-nums">
              {s.weight}%
            </span>
          </div>
        </motion.div>
      ))}

      <div className="flex items-center justify-center gap-6 pt-2">
        {[
          { label: "Low", color: "bg-spike-bull" },
          { label: "Medium", color: "bg-spike-gold" },
          { label: "High", color: "bg-spike-bear" },
        ].map((l) => (
          <div key={l.label} className="flex items-center gap-1.5">
            <div className={`w-2 h-2 rounded-full ${l.color}`} />
            <span className="text-[10px] text-spike-muted">{l.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function RegimeIndicator() {
  const phases = [
    { label: "Bearish", color: "#EF4444" },
    { label: "Consolidation", color: "#C9A96E" },
    { label: "Bullish", color: "#22C55E" },
    { label: "Euphoric", color: "#7DCEA0" },
  ];
  const activeIdx = 2;

  return (
    <div className="h-full flex flex-col justify-center">
      <div className="space-y-3 mb-6">
        {phases.map((phase, i) => (
          <motion.div
            key={phase.label}
            className="flex items-center gap-3"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: i * 0.15 }}
          >
            <span
              className={`text-xs w-24 text-right ${
                i === activeIdx
                  ? "font-semibold text-spike-cream"
                  : "text-spike-muted/40"
              }`}
            >
              {phase.label}
            </span>
            <div className="flex-1 h-2 rounded-full bg-white/[0.04] overflow-hidden">
              <motion.div
                className="h-full rounded-full"
                style={{ backgroundColor: i <= activeIdx ? phase.color : "transparent" }}
                initial={{ width: "0%" }}
                animate={{
                  width: i <= activeIdx ? `${100 - i * 15}%` : "0%",
                }}
                transition={{ delay: i * 0.15, duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
              />
            </div>
          </motion.div>
        ))}
      </div>

      <motion.div
        className="flex items-center justify-center gap-2"
        animate={{ opacity: [0.5, 1, 0.5] }}
        transition={{ duration: 2, repeat: Infinity }}
      >
        <div className="w-2 h-2 rounded-full bg-spike-bull" />
        <span className="text-sm font-medium text-spike-bull">
          Bullish Regime Active
        </span>
      </motion.div>
    </div>
  );
}

function AlertFeed() {
  const alerts = [
    {
      icon: TrendingUp,
      text: "TATAELXSI breakout — crossed 50-DMA with 2.3x volume",
      type: "bull" as const,
      ago: "2m ago",
    },
    {
      icon: Shield,
      text: "Portfolio IT concentration at 64% — above 50% threshold",
      type: "warn" as const,
      ago: "18m ago",
    },
    {
      icon: TrendingDown,
      text: "VEDL down 3.7%, nearing ₹285 stop-loss level",
      type: "bear" as const,
      ago: "1h ago",
    },
    {
      icon: Brain,
      text: "Regime shift detected: consolidation → bullish",
      type: "info" as const,
      ago: "2h ago",
    },
  ];

  const styles = {
    bull: "bg-spike-bull/10 text-spike-bull",
    warn: "bg-spike-gold/10 text-spike-gold",
    bear: "bg-spike-bear/10 text-spike-bear",
    info: "bg-spike-sage/10 text-spike-sage",
  };

  return (
    <div className="h-full flex flex-col justify-center space-y-3">
      {alerts.map((a, i) => (
        <motion.div
          key={i}
          className="flex items-start gap-3 p-3.5 rounded-2xl bg-white/[0.02] hover:bg-white/[0.04] transition-colors"
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: i * 0.12, duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
        >
          <div
            className={`w-8 h-8 rounded-xl flex items-center justify-center flex-shrink-0 ${styles[a.type]}`}
          >
            <a.icon className="w-4 h-4" />
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm text-spike-cream/85 leading-relaxed">
              {a.text}
            </p>
            <p className="text-[10px] text-spike-muted/40 mt-1">{a.ago}</p>
          </div>
        </motion.div>
      ))}
    </div>
  );
}

function LegendAgentChat() {
  const messages = [
    {
      agent: "Buffett",
      color: "from-amber-500/80 to-amber-600/80",
      text: "HDFCBANK has a durable competitive moat. At 2.8x book value, I'd accumulate on any dip below ₹1,600.",
    },
    {
      agent: "Lynch",
      color: "from-blue-500/80 to-blue-600/80",
      text: "PEG ratio of 0.9 tells me this is a growth story hiding in a bank's clothing. Classic ten-bagger potential.",
    },
    {
      agent: "Dalio",
      color: "from-spike-sage to-spike-accent",
      text: "In the current regime, private sector banks benefit from credit expansion. Risk-parity favors this position.",
    },
  ];

  return (
    <div className="h-full flex flex-col justify-center space-y-3">
      {messages.map((m, i) => (
        <motion.div
          key={m.agent}
          className="p-4 rounded-2xl bg-white/[0.02] border border-white/[0.04] hover:border-white/[0.08] transition-colors"
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: i * 0.15 + 0.2, duration: 0.5 }}
        >
          <div className="flex items-center gap-2.5 mb-2.5">
            <div
              className={`w-7 h-7 rounded-full bg-gradient-to-br ${m.color} flex items-center justify-center`}
            >
              <span className="text-[10px] font-bold text-white">
                {m.agent[0]}
              </span>
            </div>
            <span className="text-xs font-semibold text-spike-cream/70">
              {m.agent} Agent
            </span>
          </div>
          <p className="text-sm text-spike-cream/70 leading-relaxed">
            &ldquo;{m.text}&rdquo;
          </p>
        </motion.div>
      ))}
    </div>
  );
}

function ThemeBaskets() {
  const themes = [
    {
      name: "India 2030",
      stocks: 12,
      ret: "+34.2%",
      gradient: "from-spike-sage to-spike-accent",
      tags: ["Infrastructure", "Defense", "Digital"],
    },
    {
      name: "Quality Compounders",
      stocks: 8,
      ret: "+28.7%",
      gradient: "from-spike-accent to-spike-mint",
      tags: ["ROCE >20%", "Low Debt", "Consistent"],
    },
    {
      name: "Clean Energy",
      stocks: 15,
      ret: "+19.1%",
      gradient: "from-spike-gold to-spike-sage",
      tags: ["Solar", "EV", "Green H2"],
    },
  ];

  return (
    <div className="h-full flex flex-col justify-center space-y-3">
      {themes.map((t, i) => (
        <motion.div
          key={t.name}
          className="group/theme p-4 rounded-2xl bg-white/[0.02] hover:bg-white/[0.05] transition-all cursor-pointer border border-transparent hover:border-white/[0.06]"
          initial={{ opacity: 0, x: -15 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: i * 0.1, duration: 0.5 }}
        >
          <div className="flex items-center justify-between mb-2.5">
            <div className="flex items-center gap-3">
              <div
                className={`w-9 h-9 rounded-xl bg-gradient-to-br ${t.gradient} flex items-center justify-center`}
              >
                <Layers className="w-4 h-4 text-spike-dark" />
              </div>
              <div>
                <p className="text-sm font-semibold text-spike-cream">
                  {t.name}
                </p>
                <p className="text-[10px] text-spike-muted">
                  {t.stocks} stocks
                </p>
              </div>
            </div>
            <span className="text-base font-bold text-spike-bull tabular-nums">
              {t.ret}
            </span>
          </div>
          <div className="flex items-center gap-1.5">
            {t.tags.map((tag) => (
              <span
                key={tag}
                className="text-[9px] px-2 py-0.5 rounded-full bg-white/[0.04] text-spike-muted/60"
              >
                {tag}
              </span>
            ))}
          </div>
        </motion.div>
      ))}
    </div>
  );
}

function CommentaryPreview() {
  return (
    <div className="h-full flex flex-col justify-center">
      <div className="p-5 rounded-2xl bg-white/[0.03] border border-white/[0.04]">
        <div className="flex items-center gap-2.5 mb-4">
          <motion.div
            className="w-2.5 h-2.5 rounded-full bg-spike-bull"
            animate={{ opacity: [1, 0.3, 1] }}
            transition={{ duration: 1.5, repeat: Infinity }}
          />
          <span className="text-xs text-spike-muted uppercase tracking-wider">
            Live · HDFCBANK
          </span>
          <span className="text-[10px] text-spike-muted/40 ml-auto">
            3m ago
          </span>
        </div>
        <p className="text-[15px] text-spike-cream/85 leading-relaxed mb-4">
          &ldquo;Q3 results beat street estimates by 8%. NII growth at 24% YoY
          signals strong retail franchise momentum. Institutional flows
          turning positive post-results.&rdquo;
        </p>
        <div className="flex items-center gap-2">
          <span className="text-[10px] px-2.5 py-1 rounded-full bg-spike-sage/10 text-spike-sage">
            Earnings
          </span>
          <span className="text-[10px] px-2.5 py-1 rounded-full bg-spike-bull/10 text-spike-bull">
            Positive Sentiment
          </span>
          <span className="text-[10px] px-2.5 py-1 rounded-full bg-spike-accent/10 text-spike-accent">
            FII Buying
          </span>
        </div>
      </div>
    </div>
  );
}

function EducationPreview() {
  return (
    <div className="h-full flex flex-col justify-center">
      <div className="space-y-4">
        <div className="p-4 rounded-2xl bg-white/[0.03] border-l-2 border-spike-sage">
          <p className="text-[10px] text-spike-sage font-semibold uppercase tracking-wider mb-2">
            Why this matters
          </p>
          <p className="text-sm text-spike-cream/80 leading-relaxed">
            A PEG ratio below 1.0 suggests the stock is undervalued relative to
            its growth rate — it&apos;s growing faster than its price implies.
          </p>
        </div>

        <div className="p-4 rounded-2xl bg-white/[0.03] border-l-2 border-spike-accent">
          <p className="text-[10px] text-spike-accent font-semibold uppercase tracking-wider mb-2">
            What you should know
          </p>
          <p className="text-sm text-spike-cream/80 leading-relaxed">
            Concentration above 50% in one sector increases drawdown risk during
            sector rotations.
          </p>
        </div>
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────
   BENTO CARD WRAPPER
   Clean, spacious card with hover effects
   ───────────────────────────────────────────── */

function BentoCard({
  icon: Icon,
  label,
  description,
  className = "",
  gradient,
  children,
  delay = 0,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  description: string;
  className?: string;
  gradient: string;
  children: React.ReactNode;
  delay?: number;
}) {
  return (
    <Reveal delay={delay}>
      <motion.div
        className={`group relative rounded-[2rem] glass-subtle overflow-hidden h-full ${className}`}
        whileHover={{ y: -5 }}
        transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
      >
        {/* Hover gradient wash */}
        <div
          className={`absolute inset-0 bg-gradient-to-br ${gradient} opacity-0 group-hover:opacity-100 transition-opacity duration-700`}
        />

        {/* Hover border */}
        <div className="absolute inset-0 rounded-[2rem] opacity-0 group-hover:opacity-100 transition-opacity duration-500">
          <div className="absolute inset-0 rounded-[2rem] gradient-border" />
        </div>

        <div className="relative z-10 p-7 md:p-8 flex flex-col h-full">
          {/* 1. Feature name */}
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-xl bg-white/[0.06] flex items-center justify-center">
                <Icon className="w-4 h-4 text-spike-sage" />
              </div>
              <h3 className="text-base font-semibold text-spike-cream">
                {label}
              </h3>
            </div>
            <ArrowUpRight className="w-4 h-4 text-spike-muted/20 group-hover:text-spike-sage group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-all duration-300" />
          </div>

          {/* 2. Description */}
          <p className="text-sm text-spike-muted leading-relaxed mb-5 pl-[42px]">
            {description}
          </p>

          {/* Divider */}
          <div className="h-px bg-white/[0.04] mb-5" />

          {/* 3. Micro-experience — fills remaining space */}
          <div className="flex-1 min-h-0">{children}</div>
        </div>
      </motion.div>
    </Reveal>
  );
}

/* ─────────────────────────────────────────────
   MAIN FEATURES SECTION
   Asymmetric bento — visuals are the heroes
   ───────────────────────────────────────────── */

export function Features() {
  return (
    <section id="features" className="relative py-32 px-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="max-w-3xl mx-auto text-center mb-20">
          <Reveal>
            <p className="text-sm uppercase tracking-[0.2em] text-spike-sage mb-4">
              Everything you need
            </p>
          </Reveal>
          <Reveal delay={0.1}>
            <h2 className="text-4xl md:text-5xl lg:text-6xl font-bold text-spike-cream leading-tight">
              One platform.
              <span className="text-gradient"> Every dimension</span>
              <br />
              of your wealth.
            </h2>
          </Reveal>
          <Reveal delay={0.2}>
            <p className="mt-6 text-lg text-spike-muted leading-relaxed">
              From the moment you discover a stock to the day you exit — SPIKE is
              there. Researching, analyzing, monitoring, alerting, educating, and
              optimizing.
            </p>
          </Reveal>
        </div>

        {/* ── Row 1: Research (wide) + FinScore (square) ── */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          <div className="md:col-span-2">
            <BentoCard
              icon={Compass}
              label="Deep Research"
              description="5,000+ stocks analyzed across fundamentals, technicals, sentiment, and institutional flows. You ask — SPIKE already has the answer."
              gradient="from-spike-sage/8 to-transparent"
              delay={0}
            >
              <LiveTickerStrip />
            </BentoCard>
          </div>

          <div className="md:col-span-1">
            <BentoCard
              icon={Gauge}
              label="FinScore"
              description="One composite number per stock — blending fundamentals, technicals, sentiment, and flows into a single actionable score."
              gradient="from-spike-gold/8 to-transparent"
              delay={0.06}
            >
              <FinScoreGauge />
            </BentoCard>
          </div>
        </div>

        {/* ── Row 2: Strategy (wide, tall) ── */}
        <div className="grid grid-cols-1 gap-4 mb-4">
          <BentoCard
            icon={Brain}
            label="AI Strategy Engine"
            description="Describe what you want in plain English — 'defensive portfolio for retirement' or 'high-growth small caps' — and get a backtested, deployable strategy."
            gradient="from-spike-accent/8 to-transparent"
            delay={0.08}
          >
            <StrategyPreview />
          </BentoCard>
        </div>

        {/* ── Row 3: Risk + Regime + Alerts (thirds) ── */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          <BentoCard
            icon={Shield}
            label="Risk Intelligence"
            description="Real-time risk scoring, drawdown alerts, concentration warnings, and VaR calculations for every holding."
            gradient="from-spike-bear/5 to-transparent"
            delay={0.1}
          >
            <RiskHeatmap />
          </BentoCard>

          <BentoCard
            icon={BarChart3}
            label="Portfolio Command"
            description="Allocation, performance, sector exposure, and regime fit — all in one view with auto-rebalancing."
            gradient="from-spike-sage/8 to-transparent"
            delay={0.12}
          >
            <RegimeIndicator />
          </BentoCard>

          <BentoCard
            icon={Bell}
            label="Proactive Alerts"
            description="Breakouts, earnings surprises, regime shifts — you know before the crowd does."
            gradient="from-spike-gold/8 to-transparent"
            delay={0.14}
          >
            <AlertFeed />
          </BentoCard>
        </div>

        {/* ── Row 4: Legend Agents (wide) + Commentary ── */}
        <div className="grid grid-cols-1 md:grid-cols-5 gap-4 mb-4">
          <div className="md:col-span-3">
            <BentoCard
              icon={Bot}
              label="Legend Agents"
              description="Get a second opinion from AI trained on Buffett, Lynch, and Dalio. Each analyzes your picks through their lens."
              gradient="from-spike-muted/5 to-transparent"
              delay={0.16}
            >
              <LegendAgentChat />
            </BentoCard>
          </div>

          <div className="md:col-span-2">
            <BentoCard
              icon={MessageCircle}
              label="Live Commentary"
              description="Real-time analysis on your holdings — why something moved, what it means, what to consider next."
              gradient="from-spike-accent/8 to-transparent"
              delay={0.18}
            >
              <CommentaryPreview />
            </BentoCard>
          </div>
        </div>

        {/* ── Row 5: Themes (wide) + Education ── */}
        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
          <div className="md:col-span-3">
            <BentoCard
              icon={Layers}
              label="Smart Themes"
              description="Invest in narratives — 'India 2030', 'Quality Compounders', 'Clean Energy'. AI-curated baskets with continuous optimization."
              gradient="from-spike-sage/8 to-transparent"
              delay={0.2}
            >
              <ThemeBaskets />
            </BentoCard>
          </div>

          <div className="md:col-span-2">
            <BentoCard
              icon={BookOpen}
              label="Financial Education"
              description="Contextual learning woven into every interaction. Understand the 'why' behind every recommendation."
              gradient="from-spike-mint/8 to-transparent"
              delay={0.22}
            >
              <EducationPreview />
            </BentoCard>
          </div>
        </div>
      </div>
    </section>
  );
}
