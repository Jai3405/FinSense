"use client";

import { useState } from "react";
import {
  Zap,
  Play,
  Pause,
  Settings,
  TrendingUp,
  TrendingDown,
  Shield,
  Target,
  Clock,
  AlertTriangle,
  CheckCircle2,
  Activity,
  ArrowUpRight,
  ArrowDownRight,
  BarChart3,
  Lock,
  Sparkles,
  ChevronRight,
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
  loss: "#F45B69",
  lossBg: "#FEF0F1",
  warning: "#F5A623",
  warningBg: "#FFF8E6",
  pro: "#7C3AED",
  proBg: "#F3E8FF",
};

interface AutopilotAction {
  id: string;
  type: "buy" | "sell" | "rebalance" | "alert";
  symbol?: string;
  amount?: number;
  reason: string;
  timestamp: string;
  status: "executed" | "pending" | "cancelled";
}

const recentActions: AutopilotAction[] = [
  { id: "1", type: "buy", symbol: "HDFCBANK", amount: 15000, reason: "FinScore improved to 92, adding to position", timestamp: "2 hours ago", status: "executed" },
  { id: "2", type: "sell", symbol: "PAYTM", amount: 8500, reason: "Stop loss triggered at -8% from entry", timestamp: "5 hours ago", status: "executed" },
  { id: "3", type: "rebalance", reason: "IT sector overweight at 35%, rebalancing to target 25%", timestamp: "1 day ago", status: "executed" },
  { id: "4", type: "buy", symbol: "TATAPOWER", amount: 12000, reason: "Green energy theme momentum detected", timestamp: "2 days ago", status: "executed" },
  { id: "5", type: "alert", reason: "Market volatility spike - reduced exposure by 10%", timestamp: "3 days ago", status: "executed" },
];

const autopilotStats = {
  isActive: true,
  totalValue: 1247850,
  invested: 1050000,
  returns: 197850,
  returnsPercent: 18.84,
  riskScore: 45,
  actionsThisMonth: 12,
  successRate: 85,
  lastAction: "2 hours ago",
};

const riskSettings = [
  { label: "Conservative", value: 25, description: "Lower risk, stable returns" },
  { label: "Moderate", value: 50, description: "Balanced risk-reward" },
  { label: "Aggressive", value: 75, description: "Higher risk, growth focus" },
];

export default function AutopilotPage() {
  const [isActive, setIsActive] = useState(autopilotStats.isActive);
  const [riskLevel, setRiskLevel] = useState(50);

  return (
    <div className="max-w-6xl">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div
              className="w-10 h-10 rounded-xl flex items-center justify-center"
              style={{ background: `linear-gradient(135deg, ${colors.pro}, ${colors.accent})` }}
            >
              <Zap className="w-5 h-5 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-xl font-semibold" style={{ color: colors.textPrimary }}>
                  Autopilot
                </h1>
                <span
                  className="px-2 py-0.5 text-xs font-semibold rounded"
                  style={{ backgroundColor: colors.proBg, color: colors.pro }}
                >
                  PRO
                </span>
              </div>
              <p className="text-sm" style={{ color: colors.textMuted }}>
                AI-powered autonomous portfolio management
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={() => setIsActive(!isActive)}
              className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-medium transition-all"
              style={{
                backgroundColor: isActive ? colors.lossBg : colors.gainBg,
                color: isActive ? colors.loss : colors.gain,
              }}
            >
              {isActive ? (
                <>
                  <Pause className="w-4 h-4" />
                  Pause Autopilot
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  Activate Autopilot
                </>
              )}
            </button>
            <button
              className="p-2.5 rounded-xl"
              style={{ border: `1px solid ${colors.border}`, color: colors.textMuted }}
            >
              <Settings className="w-5 h-5" />
            </button>
          </div>
        </div>
      </div>

      {/* Status Banner */}
      <div
        className="rounded-xl p-4 mb-6 flex items-center justify-between"
        style={{
          backgroundColor: isActive ? colors.gainBg : colors.bgMint,
          border: `1px solid ${isActive ? colors.gain : colors.border}`,
        }}
      >
        <div className="flex items-center gap-3">
          {isActive ? (
            <Activity className="w-5 h-5" style={{ color: colors.gain }} />
          ) : (
            <Pause className="w-5 h-5" style={{ color: colors.textMuted }} />
          )}
          <div>
            <p className="text-sm font-medium" style={{ color: isActive ? colors.gain : colors.textPrimary }}>
              {isActive ? "Autopilot is actively managing your portfolio" : "Autopilot is paused"}
            </p>
            <p className="text-xs" style={{ color: colors.textMuted }}>
              Last action: {autopilotStats.lastAction}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs" style={{ color: colors.textMuted }}>
            {autopilotStats.actionsThisMonth} actions this month
          </span>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-6">
        {/* Main Stats */}
        <div className="col-span-2 space-y-6">
          {/* Portfolio Overview */}
          <div
            className="rounded-xl p-6"
            style={{ backgroundColor: colors.bg, border: `1px solid ${colors.border}` }}
          >
            <h2 className="text-sm font-semibold mb-4" style={{ color: colors.textPrimary }}>
              Autopilot Performance
            </h2>
            <div className="grid grid-cols-4 gap-6">
              <div>
                <p className="text-xs" style={{ color: colors.textMuted }}>Portfolio Value</p>
                <p className="text-xl font-semibold" style={{ color: colors.textPrimary }}>
                  ₹{autopilotStats.totalValue.toLocaleString("en-IN")}
                </p>
              </div>
              <div>
                <p className="text-xs" style={{ color: colors.textMuted }}>Total Returns</p>
                <div className="flex items-center gap-2">
                  <p className="text-xl font-semibold" style={{ color: colors.gain }}>
                    +{autopilotStats.returnsPercent}%
                  </p>
                  <ArrowUpRight className="w-4 h-4" style={{ color: colors.gain }} />
                </div>
              </div>
              <div>
                <p className="text-xs" style={{ color: colors.textMuted }}>Success Rate</p>
                <p className="text-xl font-semibold" style={{ color: colors.textPrimary }}>
                  {autopilotStats.successRate}%
                </p>
              </div>
              <div>
                <p className="text-xs" style={{ color: colors.textMuted }}>Risk Score</p>
                <div className="flex items-center gap-2">
                  <p className="text-xl font-semibold" style={{ color: colors.textPrimary }}>
                    {autopilotStats.riskScore}/100
                  </p>
                  <Shield className="w-4 h-4" style={{ color: colors.accent }} />
                </div>
              </div>
            </div>

            {/* Performance Chart Placeholder */}
            <div
              className="mt-6 h-48 rounded-xl flex items-center justify-center"
              style={{ backgroundColor: colors.bgMint }}
            >
              <div className="text-center">
                <BarChart3 className="w-12 h-12 mx-auto mb-2" style={{ color: colors.accent }} />
                <p className="text-sm" style={{ color: colors.textMuted }}>
                  Performance vs Benchmark Chart
                </p>
              </div>
            </div>
          </div>

          {/* Recent Actions */}
          <div
            className="rounded-xl overflow-hidden"
            style={{ backgroundColor: colors.bg, border: `1px solid ${colors.border}` }}
          >
            <div className="px-5 py-4 flex items-center justify-between" style={{ borderBottom: `1px solid ${colors.border}` }}>
              <h2 className="text-sm font-semibold" style={{ color: colors.textPrimary }}>
                Recent Actions
              </h2>
              <button className="text-xs font-medium" style={{ color: colors.accent }}>
                View All
              </button>
            </div>
            <div>
              {recentActions.map((action, index) => {
                const getActionIcon = () => {
                  switch (action.type) {
                    case "buy": return <ArrowUpRight className="w-4 h-4" style={{ color: colors.gain }} />;
                    case "sell": return <ArrowDownRight className="w-4 h-4" style={{ color: colors.loss }} />;
                    case "rebalance": return <Activity className="w-4 h-4" style={{ color: colors.accent }} />;
                    case "alert": return <AlertTriangle className="w-4 h-4" style={{ color: colors.warning }} />;
                  }
                };

                const getActionBg = () => {
                  switch (action.type) {
                    case "buy": return colors.gainBg;
                    case "sell": return colors.lossBg;
                    case "rebalance": return colors.bgMint;
                    case "alert": return colors.warningBg;
                  }
                };

                return (
                  <div
                    key={action.id}
                    className="px-5 py-4 flex items-center justify-between"
                    style={{ borderBottom: index < recentActions.length - 1 ? `1px solid ${colors.border}` : "none" }}
                  >
                    <div className="flex items-center gap-4">
                      <div
                        className="w-10 h-10 rounded-xl flex items-center justify-center"
                        style={{ backgroundColor: getActionBg() }}
                      >
                        {getActionIcon()}
                      </div>
                      <div>
                        <div className="flex items-center gap-2">
                          {action.symbol && (
                            <span className="text-sm font-semibold" style={{ color: colors.textPrimary }}>
                              {action.type.toUpperCase()} {action.symbol}
                            </span>
                          )}
                          {action.amount && (
                            <span className="text-sm" style={{ color: colors.textSecondary }}>
                              ₹{action.amount.toLocaleString()}
                            </span>
                          )}
                          {!action.symbol && (
                            <span className="text-sm font-semibold" style={{ color: colors.textPrimary }}>
                              {action.type.charAt(0).toUpperCase() + action.type.slice(1)}
                            </span>
                          )}
                        </div>
                        <p className="text-xs" style={{ color: colors.textMuted }}>
                          {action.reason}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-3">
                      <span className="text-xs" style={{ color: colors.textMuted }}>
                        {action.timestamp}
                      </span>
                      <CheckCircle2 className="w-4 h-4" style={{ color: colors.gain }} />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Right Sidebar */}
        <div className="space-y-6">
          {/* Risk Settings */}
          <div
            className="rounded-xl p-5"
            style={{ backgroundColor: colors.bg, border: `1px solid ${colors.border}` }}
          >
            <div className="flex items-center gap-2 mb-4">
              <Shield className="w-4 h-4" style={{ color: colors.accent }} />
              <h3 className="text-sm font-semibold" style={{ color: colors.textPrimary }}>
                Risk Level
              </h3>
            </div>
            <div className="space-y-2">
              {riskSettings.map((setting) => (
                <button
                  key={setting.label}
                  onClick={() => setRiskLevel(setting.value)}
                  className="w-full flex items-center justify-between p-3 rounded-xl transition-colors"
                  style={{
                    backgroundColor: riskLevel === setting.value ? colors.bgMint : "transparent",
                    border: `1px solid ${riskLevel === setting.value ? colors.accent : colors.border}`,
                  }}
                >
                  <div className="text-left">
                    <p className="text-sm font-medium" style={{ color: colors.textPrimary }}>
                      {setting.label}
                    </p>
                    <p className="text-xs" style={{ color: colors.textMuted }}>
                      {setting.description}
                    </p>
                  </div>
                  {riskLevel === setting.value && (
                    <CheckCircle2 className="w-5 h-5" style={{ color: colors.accent }} />
                  )}
                </button>
              ))}
            </div>
          </div>

          {/* Autopilot Rules */}
          <div
            className="rounded-xl p-5"
            style={{ backgroundColor: colors.bg, border: `1px solid ${colors.border}` }}
          >
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <Target className="w-4 h-4" style={{ color: colors.accent }} />
                <h3 className="text-sm font-semibold" style={{ color: colors.textPrimary }}>
                  Active Rules
                </h3>
              </div>
              <button className="text-xs font-medium" style={{ color: colors.accent }}>
                Edit
              </button>
            </div>
            <div className="space-y-3">
              {[
                { rule: "Stop Loss", value: "-8%", active: true },
                { rule: "Take Profit", value: "+25%", active: true },
                { rule: "Max Position Size", value: "15%", active: true },
                { rule: "Rebalance Threshold", value: "10%", active: true },
                { rule: "Sector Limit", value: "30%", active: false },
              ].map((item) => (
                <div key={item.rule} className="flex items-center justify-between">
                  <span className="text-sm" style={{ color: item.active ? colors.textSecondary : colors.textMuted }}>
                    {item.rule}
                  </span>
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium" style={{ color: item.active ? colors.textPrimary : colors.textMuted }}>
                      {item.value}
                    </span>
                    <div
                      className="w-2 h-2 rounded-full"
                      style={{ backgroundColor: item.active ? colors.gain : colors.border }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Pro Features */}
          <div
            className="rounded-xl p-5"
            style={{ background: `linear-gradient(135deg, ${colors.pro}15, ${colors.accent}10)`, border: `1px solid ${colors.pro}30` }}
          >
            <div className="flex items-center gap-2 mb-3">
              <Sparkles className="w-4 h-4" style={{ color: colors.pro }} />
              <h3 className="text-sm font-semibold" style={{ color: colors.textPrimary }}>
                Pro Features
              </h3>
            </div>
            <ul className="space-y-2">
              {[
                "Unlimited automated trades",
                "Advanced risk controls",
                "Priority execution",
                "Tax-loss harvesting",
                "Custom strategies",
              ].map((feature) => (
                <li key={feature} className="flex items-center gap-2 text-xs" style={{ color: colors.textSecondary }}>
                  <CheckCircle2 className="w-3.5 h-3.5" style={{ color: colors.pro }} />
                  {feature}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
