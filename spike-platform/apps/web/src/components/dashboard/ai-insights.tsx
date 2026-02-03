"use client";

import {
  Brain,
  AlertTriangle,
  TrendingUp,
  Lightbulb,
  ChevronRight,
  Sparkles,
} from "lucide-react";
import { cn } from "@/lib/utils";

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

const insightColors = {
  opportunity: "text-spike-bull bg-spike-bull/20",
  warning: "text-spike-warning bg-spike-warning/20",
  trend: "text-spike-accent bg-spike-accent/20",
  idea: "text-spike-secondary bg-spike-secondary/20",
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
    <div className="rounded-2xl bg-white/5 border border-white/10 p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-spike-gradient flex items-center justify-center">
            <Sparkles className="w-4 h-4 text-white" />
          </div>
          <h2 className="text-lg font-semibold text-white">AI Insights</h2>
        </div>
        <span className="px-2 py-1 text-xs rounded-full bg-spike-primary/20 text-spike-primary">
          3 new
        </span>
      </div>

      <div className="space-y-4">
        {insights.map((insight, index) => {
          const Icon = insightIcons[insight.type];
          return (
            <div
              key={index}
              className="p-4 rounded-xl bg-white/5 hover:bg-white/10 transition cursor-pointer group"
            >
              <div className="flex items-start gap-3">
                <div
                  className={cn(
                    "w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0",
                    insightColors[insight.type]
                  )}
                >
                  <Icon className="w-4 h-4" />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between gap-2 mb-1">
                    <p className="font-medium text-white text-sm truncate">
                      {insight.title}
                    </p>
                    <span className="text-xs text-slate-500 flex-shrink-0">
                      {insight.timestamp}
                    </span>
                  </div>
                  <p className="text-sm text-slate-400 line-clamp-2">
                    {insight.description}
                  </p>
                  {insight.action && (
                    <button className="flex items-center gap-1 mt-2 text-sm text-spike-primary hover:text-spike-primary/80 transition">
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

      <button className="w-full mt-4 py-2.5 rounded-xl bg-spike-gradient text-white text-sm font-medium hover:opacity-90 transition flex items-center justify-center gap-2">
        <Brain className="w-4 h-4" />
        Ask AI Anything
      </button>
    </div>
  );
}
