"use client";

import { useState } from "react";
import {
  User,
  Bell,
  Shield,
  Palette,
  CreditCard,
  Link,
  HelpCircle,
  LogOut,
  ChevronRight,
  Check,
  Moon,
  Sun,
  Smartphone,
  Mail,
  MessageSquare,
  AlertTriangle,
  TrendingUp,
  Lock,
  Key,
  Building2,
  ExternalLink,
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
};

const settingsTabs = [
  { id: "profile", label: "Profile", icon: User },
  { id: "notifications", label: "Notifications", icon: Bell },
  { id: "security", label: "Security", icon: Shield },
  { id: "appearance", label: "Appearance", icon: Palette },
  { id: "billing", label: "Billing", icon: CreditCard },
  { id: "integrations", label: "Integrations", icon: Link },
  { id: "help", label: "Help & Support", icon: HelpCircle },
];

const brokerIntegrations = [
  { name: "Zerodha", status: "connected", logo: "Z" },
  { name: "Groww", status: "disconnected", logo: "G" },
  { name: "Upstox", status: "disconnected", logo: "U" },
  { name: "Angel One", status: "disconnected", logo: "A" },
];

export default function SettingsPage() {
  const [activeTab, setActiveTab] = useState("profile");
  const [notifications, setNotifications] = useState({
    priceAlerts: true,
    portfolioUpdates: true,
    aiInsights: true,
    weeklyReport: true,
    marketNews: false,
    promotions: false,
  });

  const renderContent = () => {
    switch (activeTab) {
      case "profile":
        return (
          <div className="space-y-6">
            <div>
              <h2 className="text-lg font-semibold mb-4" style={{ color: colors.textPrimary }}>
                Profile Settings
              </h2>
              <div
                className="rounded-xl p-6"
                style={{ backgroundColor: colors.bg, border: `1px solid ${colors.border}` }}
              >
                <div className="flex items-start gap-6">
                  <div
                    className="w-20 h-20 rounded-2xl flex items-center justify-center text-2xl font-bold text-white"
                    style={{ backgroundColor: colors.accent }}
                  >
                    JD
                  </div>
                  <div className="flex-1">
                    <button className="text-sm font-medium" style={{ color: colors.accent }}>
                      Change Photo
                    </button>
                    <div className="grid grid-cols-2 gap-4 mt-4">
                      <div>
                        <label className="text-xs font-medium" style={{ color: colors.textMuted }}>
                          Full Name
                        </label>
                        <input
                          type="text"
                          defaultValue="Jay Desai"
                          className="w-full mt-1 px-3 py-2 text-sm rounded-lg focus:outline-none"
                          style={{ backgroundColor: colors.bgMint, border: `1px solid ${colors.border}`, color: colors.textPrimary }}
                        />
                      </div>
                      <div>
                        <label className="text-xs font-medium" style={{ color: colors.textMuted }}>
                          Email
                        </label>
                        <input
                          type="email"
                          defaultValue="jay@example.com"
                          className="w-full mt-1 px-3 py-2 text-sm rounded-lg focus:outline-none"
                          style={{ backgroundColor: colors.bgMint, border: `1px solid ${colors.border}`, color: colors.textPrimary }}
                        />
                      </div>
                      <div>
                        <label className="text-xs font-medium" style={{ color: colors.textMuted }}>
                          Phone
                        </label>
                        <input
                          type="tel"
                          defaultValue="+91 98765 43210"
                          className="w-full mt-1 px-3 py-2 text-sm rounded-lg focus:outline-none"
                          style={{ backgroundColor: colors.bgMint, border: `1px solid ${colors.border}`, color: colors.textPrimary }}
                        />
                      </div>
                      <div>
                        <label className="text-xs font-medium" style={{ color: colors.textMuted }}>
                          PAN Number
                        </label>
                        <input
                          type="text"
                          defaultValue="ABCDE1234F"
                          className="w-full mt-1 px-3 py-2 text-sm rounded-lg focus:outline-none"
                          style={{ backgroundColor: colors.bgMint, border: `1px solid ${colors.border}`, color: colors.textPrimary }}
                        />
                      </div>
                    </div>
                    <button
                      className="mt-4 px-4 py-2 rounded-lg text-sm font-medium text-white"
                      style={{ backgroundColor: colors.accent }}
                    >
                      Save Changes
                    </button>
                  </div>
                </div>
              </div>
            </div>

            <div>
              <h3 className="text-sm font-semibold mb-3" style={{ color: colors.textPrimary }}>
                Investment Profile
              </h3>
              <div
                className="rounded-xl p-5"
                style={{ backgroundColor: colors.bg, border: `1px solid ${colors.border}` }}
              >
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-xs font-medium" style={{ color: colors.textMuted }}>
                      Risk Appetite
                    </label>
                    <select
                      className="w-full mt-1 px-3 py-2 text-sm rounded-lg focus:outline-none"
                      style={{ backgroundColor: colors.bgMint, border: `1px solid ${colors.border}`, color: colors.textPrimary }}
                      defaultValue="moderate"
                    >
                      <option value="conservative">Conservative</option>
                      <option value="moderate">Moderate</option>
                      <option value="aggressive">Aggressive</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-xs font-medium" style={{ color: colors.textMuted }}>
                      Investment Horizon
                    </label>
                    <select
                      className="w-full mt-1 px-3 py-2 text-sm rounded-lg focus:outline-none"
                      style={{ backgroundColor: colors.bgMint, border: `1px solid ${colors.border}`, color: colors.textPrimary }}
                      defaultValue="long"
                    >
                      <option value="short">Short Term ({"<"}1 year)</option>
                      <option value="medium">Medium Term (1-3 years)</option>
                      <option value="long">Long Term ({">"}3 years)</option>
                    </select>
                  </div>
                </div>
              </div>
            </div>
          </div>
        );

      case "notifications":
        return (
          <div className="space-y-6">
            <h2 className="text-lg font-semibold" style={{ color: colors.textPrimary }}>
              Notification Preferences
            </h2>
            <div
              className="rounded-xl divide-y"
              style={{ backgroundColor: colors.bg, border: `1px solid ${colors.border}`, borderColor: colors.border }}
            >
              {[
                { key: "priceAlerts", icon: TrendingUp, label: "Price Alerts", description: "Get notified when stocks hit your target price" },
                { key: "portfolioUpdates", icon: Building2, label: "Portfolio Updates", description: "Daily summary of your portfolio performance" },
                { key: "aiInsights", icon: AlertTriangle, label: "AI Insights", description: "Receive AI-generated investment insights" },
                { key: "weeklyReport", icon: Mail, label: "Weekly Report", description: "Comprehensive weekly market and portfolio report" },
                { key: "marketNews", icon: MessageSquare, label: "Market News", description: "Breaking news affecting your holdings" },
                { key: "promotions", icon: Bell, label: "Promotions", description: "Updates about new features and offers" },
              ].map((item) => {
                const Icon = item.icon;
                return (
                  <div key={item.key} className="p-4 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div
                        className="w-10 h-10 rounded-xl flex items-center justify-center"
                        style={{ backgroundColor: colors.bgMint }}
                      >
                        <Icon className="w-5 h-5" style={{ color: colors.accent }} />
                      </div>
                      <div>
                        <p className="text-sm font-medium" style={{ color: colors.textPrimary }}>
                          {item.label}
                        </p>
                        <p className="text-xs" style={{ color: colors.textMuted }}>
                          {item.description}
                        </p>
                      </div>
                    </div>
                    <button
                      onClick={() => setNotifications({ ...notifications, [item.key]: !notifications[item.key as keyof typeof notifications] })}
                      className="relative w-11 h-6 rounded-full transition-colors"
                      style={{ backgroundColor: notifications[item.key as keyof typeof notifications] ? colors.accent : colors.border }}
                    >
                      <div
                        className="absolute top-1 w-4 h-4 rounded-full bg-white transition-transform"
                        style={{ left: notifications[item.key as keyof typeof notifications] ? "calc(100% - 20px)" : "4px" }}
                      />
                    </button>
                  </div>
                );
              })}
            </div>
          </div>
        );

      case "security":
        return (
          <div className="space-y-6">
            <h2 className="text-lg font-semibold" style={{ color: colors.textPrimary }}>
              Security Settings
            </h2>
            <div
              className="rounded-xl p-5 space-y-4"
              style={{ backgroundColor: colors.bg, border: `1px solid ${colors.border}` }}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Lock className="w-5 h-5" style={{ color: colors.accent }} />
                  <div>
                    <p className="text-sm font-medium" style={{ color: colors.textPrimary }}>Change Password</p>
                    <p className="text-xs" style={{ color: colors.textMuted }}>Last changed 30 days ago</p>
                  </div>
                </div>
                <ChevronRight className="w-5 h-5" style={{ color: colors.border }} />
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Key className="w-5 h-5" style={{ color: colors.accent }} />
                  <div>
                    <p className="text-sm font-medium" style={{ color: colors.textPrimary }}>Two-Factor Authentication</p>
                    <p className="text-xs" style={{ color: colors.gain }}>Enabled</p>
                  </div>
                </div>
                <ChevronRight className="w-5 h-5" style={{ color: colors.border }} />
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Smartphone className="w-5 h-5" style={{ color: colors.accent }} />
                  <div>
                    <p className="text-sm font-medium" style={{ color: colors.textPrimary }}>Active Sessions</p>
                    <p className="text-xs" style={{ color: colors.textMuted }}>2 devices logged in</p>
                  </div>
                </div>
                <ChevronRight className="w-5 h-5" style={{ color: colors.border }} />
              </div>
            </div>
          </div>
        );

      case "integrations":
        return (
          <div className="space-y-6">
            <h2 className="text-lg font-semibold" style={{ color: colors.textPrimary }}>
              Broker Integrations
            </h2>
            <p className="text-sm" style={{ color: colors.textMuted }}>
              Connect your brokerage accounts for seamless portfolio tracking and trading.
            </p>
            <div className="grid grid-cols-2 gap-4">
              {brokerIntegrations.map((broker) => (
                <div
                  key={broker.name}
                  className="rounded-xl p-4 flex items-center justify-between"
                  style={{ backgroundColor: colors.bg, border: `1px solid ${colors.border}` }}
                >
                  <div className="flex items-center gap-3">
                    <div
                      className="w-12 h-12 rounded-xl flex items-center justify-center text-lg font-bold"
                      style={{ backgroundColor: colors.bgMint, color: colors.accent }}
                    >
                      {broker.logo}
                    </div>
                    <div>
                      <p className="text-sm font-medium" style={{ color: colors.textPrimary }}>
                        {broker.name}
                      </p>
                      <p
                        className="text-xs"
                        style={{ color: broker.status === "connected" ? colors.gain : colors.textMuted }}
                      >
                        {broker.status === "connected" ? "Connected" : "Not connected"}
                      </p>
                    </div>
                  </div>
                  <button
                    className="px-3 py-1.5 text-xs font-medium rounded-lg"
                    style={{
                      backgroundColor: broker.status === "connected" ? colors.lossBg : colors.gainBg,
                      color: broker.status === "connected" ? colors.loss : colors.gain,
                    }}
                  >
                    {broker.status === "connected" ? "Disconnect" : "Connect"}
                  </button>
                </div>
              ))}
            </div>
          </div>
        );

      case "appearance":
        return (
          <div className="space-y-6">
            <h2 className="text-lg font-semibold" style={{ color: colors.textPrimary }}>
              Appearance
            </h2>
            <div
              className="rounded-xl p-5"
              style={{ backgroundColor: colors.bg, border: `1px solid ${colors.border}` }}
            >
              <p className="text-sm font-medium mb-3" style={{ color: colors.textPrimary }}>Theme</p>
              <div className="flex gap-3">
                {[
                  { id: "light", icon: Sun, label: "Light" },
                  { id: "dark", icon: Moon, label: "Dark" },
                  { id: "system", icon: Smartphone, label: "System" },
                ].map((theme) => {
                  const Icon = theme.icon;
                  const isActive = theme.id === "light";
                  return (
                    <button
                      key={theme.id}
                      className="flex-1 flex flex-col items-center gap-2 p-4 rounded-xl transition-colors"
                      style={{
                        backgroundColor: isActive ? colors.bgMint : "transparent",
                        border: `1px solid ${isActive ? colors.accent : colors.border}`,
                      }}
                    >
                      <Icon className="w-6 h-6" style={{ color: isActive ? colors.accent : colors.textMuted }} />
                      <span className="text-sm" style={{ color: isActive ? colors.accent : colors.textSecondary }}>
                        {theme.label}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
        );

      default:
        return (
          <div className="text-center py-12">
            <p style={{ color: colors.textMuted }}>Coming soon...</p>
          </div>
        );
    }
  };

  return (
    <div className="max-w-5xl">
      <div className="mb-6">
        <h1 className="text-xl font-semibold" style={{ color: colors.textPrimary }}>
          Settings
        </h1>
        <p className="text-sm" style={{ color: colors.textMuted }}>
          Manage your account and preferences
        </p>
      </div>

      <div className="grid grid-cols-4 gap-6">
        {/* Sidebar */}
        <div className="space-y-1">
          {settingsTabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className="w-full flex items-center gap-3 px-4 py-2.5 rounded-xl text-left transition-colors"
                style={{
                  backgroundColor: isActive ? colors.bgMint : "transparent",
                  color: isActive ? colors.accent : colors.textSecondary,
                }}
                onMouseEnter={(e) => {
                  if (!isActive) e.currentTarget.style.backgroundColor = colors.bgHover;
                }}
                onMouseLeave={(e) => {
                  if (!isActive) e.currentTarget.style.backgroundColor = "transparent";
                }}
              >
                <Icon className="w-5 h-5" />
                <span className="text-sm font-medium">{tab.label}</span>
              </button>
            );
          })}

          <div className="pt-4 mt-4" style={{ borderTop: `1px solid ${colors.border}` }}>
            <button
              className="w-full flex items-center gap-3 px-4 py-2.5 rounded-xl text-left transition-colors"
              style={{ color: colors.loss }}
              onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = colors.lossBg)}
              onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
            >
              <LogOut className="w-5 h-5" />
              <span className="text-sm font-medium">Sign Out</span>
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="col-span-3">
          {renderContent()}
        </div>
      </div>
    </div>
  );
}
