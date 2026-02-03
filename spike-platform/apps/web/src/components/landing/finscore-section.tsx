"use client";

import { useRef } from "react";
import { motion, useInView } from "framer-motion";
import {
  Activity,
  AlertTriangle,
  Brain,
  Eye,
  Gauge,
  Shield,
  TrendingUp,
} from "lucide-react";
import { Reveal } from "./reveal";

const liveInsights = [
  {
    icon: TrendingUp,
    color: "text-spike-bull",
    bg: "bg-spike-bull/10",
    title: "TATAELXSI breakout detected",
    desc: "Crossed 50-DMA with 2.3x average volume. Momentum accelerating.",
    time: "2m ago",
  },
  {
    icon: Shield,
    color: "text-spike-gold",
    bg: "bg-spike-gold/10",
    title: "Portfolio concentration alert",
    desc: "IT sector at 64% — above your 50% threshold. Consider rebalancing.",
    time: "18m ago",
  },
  {
    icon: Brain,
    color: "text-spike-sage",
    bg: "bg-spike-sage/10",
    title: "Regime shift: consolidation → bullish",
    desc: "Market conditions favor momentum strategies. Adjusting recommendations.",
    time: "1h ago",
  },
  {
    icon: Gauge,
    color: "text-spike-accent",
    bg: "bg-spike-accent/10",
    title: "HDFCBANK FinScore upgraded to 82",
    desc: "Strong Q3 earnings (+8% beat), improving technicals, and positive sentiment pushed score from 74 → 82.",
    time: "3h ago",
  },
  {
    icon: AlertTriangle,
    color: "text-spike-bear",
    bg: "bg-spike-bear/10",
    title: "Risk: VEDL approaching stop-loss",
    desc: "Down 3.7% today, 2.1% from your ₹285 alert level. Review position.",
    time: "5h ago",
  },
];

const monitoringStats = [
  { label: "Stocks monitored", value: "5,000+" },
  { label: "Signals processed / day", value: "2.4M" },
  { label: "Avg. alert latency", value: "<3 sec" },
  { label: "Data sources", value: "47" },
];

export function FinScoreSection() {
  const sectionRef = useRef<HTMLDivElement>(null);
  const isInView = useInView(sectionRef, { once: true, margin: "-100px" });

  return (
    <section id="intelligence" ref={sectionRef} className="relative py-32 px-6">
      {/* Background accent */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full bg-spike-sage/5 blur-[120px]" />
      </div>

      <div className="max-w-7xl mx-auto relative">
        {/* Header */}
        <div className="max-w-3xl mx-auto text-center mb-20">
          <Reveal>
            <p className="text-sm uppercase tracking-[0.2em] text-spike-sage mb-4">
              Always-on intelligence
            </p>
          </Reveal>
          <Reveal delay={0.1}>
            <h2 className="text-4xl md:text-5xl lg:text-6xl font-bold text-spike-cream leading-tight">
              Your money never
              <span className="text-gradient"> sleeps.</span>
              <br />
              Neither do we.
            </h2>
          </Reveal>
          <Reveal delay={0.2}>
            <p className="mt-6 text-lg text-spike-muted leading-relaxed">
              SPIKE continuously watches your portfolio, the market, and every
              stock in India. When something matters — a breakout, a risk, an
              opportunity, an earnings surprise — you know instantly.
            </p>
          </Reveal>
        </div>

        {/* Two-column layout */}
        <div className="grid lg:grid-cols-5 gap-8 items-start">
          {/* Left — Live feed (3 cols) */}
          <Reveal direction="left" className="lg:col-span-3">
            <div className="glass-subtle rounded-3xl p-6 overflow-hidden">
              {/* Feed header */}
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="relative">
                    <Activity className="w-5 h-5 text-spike-sage" />
                    <span className="absolute -top-1 -right-1 w-2 h-2 bg-spike-bull rounded-full animate-ping" />
                  </div>
                  <span className="text-sm font-medium text-spike-cream">
                    Live Intelligence Feed
                  </span>
                </div>
                <span className="text-xs text-spike-muted">
                  Personalized to your portfolio
                </span>
              </div>

              {/* Insights */}
              <div className="space-y-3">
                {liveInsights.map((insight, i) => (
                  <motion.div
                    key={i}
                    className="flex items-start gap-4 p-4 rounded-2xl bg-white/[0.02] hover:bg-white/[0.05] transition-colors cursor-pointer group"
                    initial={{ opacity: 0, x: -20 }}
                    animate={isInView ? { opacity: 1, x: 0 } : {}}
                    transition={{
                      delay: i * 0.12 + 0.3,
                      duration: 0.5,
                      ease: [0.16, 1, 0.3, 1],
                    }}
                  >
                    <div
                      className={`w-9 h-9 rounded-xl ${insight.bg} flex items-center justify-center flex-shrink-0 mt-0.5`}
                    >
                      <insight.icon className={`w-4 h-4 ${insight.color}`} />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between gap-2">
                        <p className="text-sm font-medium text-spike-cream truncate">
                          {insight.title}
                        </p>
                        <span className="text-xs text-spike-muted/50 flex-shrink-0">
                          {insight.time}
                        </span>
                      </div>
                      <p className="text-sm text-spike-muted mt-1 leading-relaxed">
                        {insight.desc}
                      </p>
                    </div>
                  </motion.div>
                ))}
              </div>

              {/* Fade at bottom */}
              <div className="h-8 bg-gradient-to-t from-spike-deep/80 to-transparent -mx-6 -mb-6 mt-2" />
            </div>
          </Reveal>

          {/* Right — Stats + Summary (2 cols) */}
          <div className="lg:col-span-2 space-y-6">
            <Reveal direction="right" delay={0.1}>
              <div className="glass-subtle rounded-3xl p-6">
                <h3 className="text-sm font-medium text-spike-muted mb-5 uppercase tracking-wider">
                  What's running right now
                </h3>
                <div className="space-y-5">
                  {monitoringStats.map((stat, i) => (
                    <motion.div
                      key={stat.label}
                      className="flex items-center justify-between"
                      initial={{ opacity: 0, y: 10 }}
                      animate={isInView ? { opacity: 1, y: 0 } : {}}
                      transition={{ delay: i * 0.1 + 0.5 }}
                    >
                      <span className="text-sm text-spike-muted">
                        {stat.label}
                      </span>
                      <span className="text-sm font-semibold text-spike-cream">
                        {stat.value}
                      </span>
                    </motion.div>
                  ))}
                </div>
              </div>
            </Reveal>

            <Reveal direction="right" delay={0.2}>
              <div className="glass-subtle rounded-3xl p-6">
                <div className="flex items-center gap-3 mb-4">
                  <Eye className="w-5 h-5 text-spike-sage" />
                  <h3 className="text-sm font-medium text-spike-cream">
                    Portfolio Health
                  </h3>
                </div>

                {/* Mini health indicators */}
                <div className="space-y-4">
                  {[
                    { label: "Diversification", score: 72, status: "Good" },
                    { label: "Risk-adjusted returns", score: 85, status: "Strong" },
                    { label: "Regime alignment", score: 91, status: "Excellent" },
                    { label: "Cost efficiency", score: 68, status: "Review" },
                  ].map((metric, i) => (
                    <div key={metric.label}>
                      <div className="flex items-center justify-between mb-1.5">
                        <span className="text-xs text-spike-muted">
                          {metric.label}
                        </span>
                        <span className="text-xs font-medium text-spike-cream">
                          {metric.status}
                        </span>
                      </div>
                      <div className="h-1 rounded-full bg-white/5 overflow-hidden">
                        <motion.div
                          className="h-full rounded-full bg-gradient-to-r from-spike-sage to-spike-accent"
                          initial={{ width: "0%" }}
                          animate={
                            isInView
                              ? { width: `${metric.score}%` }
                              : { width: "0%" }
                          }
                          transition={{
                            duration: 1,
                            delay: i * 0.1 + 0.8,
                            ease: [0.16, 1, 0.3, 1],
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </Reveal>

            <Reveal direction="right" delay={0.3}>
              <div className="glass-subtle rounded-3xl p-6">
                <div className="flex items-center gap-3 mb-4">
                  <Gauge className="w-5 h-5 text-spike-accent" />
                  <h3 className="text-sm font-medium text-spike-cream">
                    FinScore Sample
                  </h3>
                </div>

                {/* Mini FinScore card */}
                <div className="space-y-3">
                  {[
                    { ticker: "HDFCBANK", score: 82, delta: "+8", signal: "Strong Buy" },
                    { ticker: "INFY", score: 71, delta: "+2", signal: "Buy" },
                    { ticker: "VEDL", score: 38, delta: "-12", signal: "Caution" },
                  ].map((stock, i) => (
                    <motion.div
                      key={stock.ticker}
                      className="flex items-center justify-between p-3 rounded-xl bg-white/[0.03]"
                      initial={{ opacity: 0, x: 20 }}
                      animate={isInView ? { opacity: 1, x: 0 } : {}}
                      transition={{ delay: i * 0.1 + 1.0 }}
                    >
                      <div className="flex items-center gap-3">
                        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-spike-sage/20 to-spike-accent/20 flex items-center justify-center">
                          <span className="text-xs font-bold text-spike-cream">
                            {stock.score}
                          </span>
                        </div>
                        <div>
                          <p className="text-sm font-medium text-spike-cream">
                            {stock.ticker}
                          </p>
                          <p className="text-xs text-spike-muted">
                            {stock.signal}
                          </p>
                        </div>
                      </div>
                      <span
                        className={`text-xs font-medium ${
                          stock.delta.startsWith("+")
                            ? "text-spike-bull"
                            : "text-spike-bear"
                        }`}
                      >
                        {stock.delta}
                      </span>
                    </motion.div>
                  ))}
                </div>
                <p className="text-xs text-spike-muted/40 mt-3">
                  Composite score from fundamentals, technicals, sentiment &amp; flows
                </p>
              </div>
            </Reveal>

            <Reveal direction="right" delay={0.4}>
              <div className="glass-subtle rounded-3xl p-6">
                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 rounded-lg bg-spike-sage/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                    <Brain className="w-4 h-4 text-spike-sage" />
                  </div>
                  <div>
                    <p className="text-sm text-spike-cream/90 leading-relaxed">
                      &ldquo;Your portfolio is well-positioned for the current bullish
                      regime. I&apos;d suggest trimming TATASTEEL (-4.2% today) and
                      adding to HDFCBANK which just reported a strong quarter.&rdquo;
                    </p>
                    <p className="text-xs text-spike-muted/50 mt-3">
                      SPIKE AI &middot; Personalized to your holdings
                    </p>
                  </div>
                </div>
              </div>
            </Reveal>
          </div>
        </div>
      </div>
    </section>
  );
}
