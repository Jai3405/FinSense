"use client";

import {
  Brain,
  AlertTriangle,
  TrendingUp,
  Lightbulb,
  ChevronRight,
  Sparkles,
} from "lucide-react";

// Color constants
const colors = {
  bg: "#FFFFFF",
  bgHover: "#F0FBF9",
  bgMint: "#F5FFFC",
  border: "#B8DDD7",
  textPrimary: "#0D3331",
  textSecondary: "#3D6B66",
  textMuted: "#6B9B94",
  accent: "#00897B",
  gain: "#00B386",
  gainBg: "#E6F9F4",
  warning: "#F5A623",
  warningBg: "#FFF8E6",
  loss: "#F45B69",
  lossBg: "#FEF0F1",
};

interface Insight {
  type: "opportunity" | "warning" | "trend" | "idea";
  title: string;
  description: string;
  action?: string;
  timestamp: string;
}

const insightIcons = {
  opportunity: TrendingUp,
  warning: AlertTriangle,
  trend: Brain,
  idea: Lightbulb,
};

const insightStyles = {
  opportunity: { bg: colors.gainBg, color: colors.gain },
  warning: { bg: colors.warningBg, color: colors.warning },
  trend: { bg: `${colors.accent}15`, color: colors.accent },
  idea: { bg: colors.bgMint, color: colors.textSecondary },
};

export function AIInsights() {
  const insights: Insight[] = [
    {
      type: "opportunity",
      title: "TATAELXSI showing breakout pattern",
      description:
        "Stock crossed 50-DMA with 2x volume. FinScore: 8.2. Consider position.",
      action: "View Analysis",
      timestamp: "2m ago",
    },
    {
      type: "warning",
      title: "Portfolio concentrated in IT",
      description:
        "65% allocation to IT sector. Consider diversifying into Pharma or FMCG.",
      action: "Rebalance",
      timestamp: "15m ago",
    },
    {
      type: "trend",
      title: "Regime shift detected",
      description:
        "Market transitioning from consolidation to bullish phase. Adjust strategy.",
      timestamp: "1h ago",
    },
  ];

  return (
    <div
      className="rounded-2xl p-6"
      style={{ backgroundColor: colors.bg, border: `1px solid ${colors.border}` }}
    >
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-2">
          <div
            className="w-8 h-8 rounded-lg flex items-center justify-center"
            style={{ background: `linear-gradient(135deg, ${colors.accent}, ${colors.gain})` }}
          >
            <Sparkles className="w-4 h-4 text-white" />
          </div>
          <h2 className="text-lg font-semibold" style={{ color: colors.textPrimary }}>
            AI Insights
          </h2>
        </div>
        <span
          className="px-2 py-1 text-xs rounded-full"
          style={{ backgroundColor: `${colors.accent}15`, color: colors.accent }}
        >
          3 new
        </span>
      </div>

      <div className="space-y-4">
        {insights.map((insight, index) => {
          const Icon = insightIcons[insight.type];
          const style = insightStyles[insight.type];
          return (
            <div
              key={index}
              className="p-4 rounded-xl transition cursor-pointer group"
              style={{ backgroundColor: colors.bgMint, border: `1px solid ${colors.border}` }}
              onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = colors.bgHover)}
              onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = colors.bgMint)}
            >
              <div className="flex items-start gap-3">
                <div
                  className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0"
                  style={{ backgroundColor: style.bg }}
                >
                  <Icon className="w-4 h-4" style={{ color: style.color }} />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between gap-2 mb-1">
                    <p className="font-medium text-sm truncate" style={{ color: colors.textPrimary }}>
                      {insight.title}
                    </p>
                    <span className="text-xs flex-shrink-0" style={{ color: colors.textMuted }}>
                      {insight.timestamp}
                    </span>
                  </div>
                  <p className="text-sm line-clamp-2" style={{ color: colors.textSecondary }}>
                    {insight.description}
                  </p>
                  {insight.action && (
                    <button
                      className="flex items-center gap-1 mt-2 text-sm transition-colors"
                      style={{ color: colors.accent }}
                      onMouseEnter={(e) => (e.currentTarget.style.opacity = "0.8")}
                      onMouseLeave={(e) => (e.currentTarget.style.opacity = "1")}
                    >
                      {insight.action}
                      <ChevronRight className="w-4 h-4" />
                    </button>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <button
        className="w-full mt-4 py-2.5 rounded-xl text-white text-sm font-medium transition-opacity flex items-center justify-center gap-2"
        style={{ background: `linear-gradient(135deg, ${colors.accent}, ${colors.gain})` }}
        onMouseEnter={(e) => (e.currentTarget.style.opacity = "0.9")}
        onMouseLeave={(e) => (e.currentTarget.style.opacity = "1")}
      >
        <Brain className="w-4 h-4" />
        Ask AI Anything
      </button>
    </div>
  );
}
