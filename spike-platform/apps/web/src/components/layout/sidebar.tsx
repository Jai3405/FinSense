"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { useSidebar } from "@/components/providers/sidebar-provider";
import {
  BarChart3,
  Bot,
  Briefcase,
  Compass,
  Home,
  Layers,
  LineChart,
  Settings,
  Sparkles,
  Target,
  Zap,
  ChevronLeft,
  ChevronRight,
  TrendingUp,
  TrendingDown,
} from "lucide-react";
import { SpikeLogo } from "@/components/ui/spike-logo";

// Color constants
const colors = {
  bg: "#FFFFFF",
  bgHover: "#F0FBF9",
  bgActive: "#E5F7F4",
  border: "#B8DDD7",
  textPrimary: "#0D3331",
  textSecondary: "#3D6B66",
  textMuted: "#6B9B94",
  accent: "#00897B",
  accentLight: "#E0F2F1",
  gain: "#00B386",
  gainBg: "#E6F9F4",
  loss: "#F45B69",
};

const navigation = [
  { name: "Dashboard", href: "/dashboard", icon: Home },
  { name: "Discover", href: "/dashboard/discover", icon: Compass },
  { name: "FinScore", href: "/dashboard/finscore", icon: Target },
  { name: "Portfolio", href: "/dashboard/portfolio", icon: Briefcase },
  { name: "Themes", href: "/dashboard/themes", icon: Layers },
  { name: "Screener", href: "/dashboard/screener", icon: BarChart3 },
  { name: "Charts", href: "/dashboard/charts", icon: LineChart },
];

const aiTools = [
  { name: "Strategy-GPT", href: "/dashboard/strategy", icon: Sparkles },
  { name: "Legend Agents", href: "/dashboard/legends", icon: Bot },
  { name: "Autopilot", href: "/dashboard/autopilot", icon: Zap, badge: "Pro" },
];

// Mini portfolio widget data (mock)
const portfolioData = {
  value: "₹12.47L",
  change: 2.42,
  isPositive: true,
};

// Market status (mock)
const marketStatus = {
  isOpen: true,
  nifty: { value: "22,150", change: 0.57 },
};

export function Sidebar() {
  const pathname = usePathname();
  const { isCollapsed, setIsCollapsed, isHovering, setIsHovering } = useSidebar();

  const showExpanded = !isCollapsed || isHovering;

  return (
    <>
      <aside
        className={cn(
          "fixed inset-y-0 left-0 z-50 hidden lg:flex flex-col transition-all duration-300 ease-out",
          showExpanded ? "w-64" : "w-[72px]"
        )}
        style={{
          backgroundColor: colors.bg,
          borderRight: `1px solid ${colors.border}`,
        }}
        onMouseEnter={() => isCollapsed && setIsHovering(true)}
        onMouseLeave={() => setIsHovering(false)}
      >
        {/* Logo Section */}
        <div
          className={cn(
            "flex h-16 items-center transition-all duration-300",
            showExpanded ? "justify-between px-4" : "justify-center px-0"
          )}
          style={{ borderBottom: `1px solid ${colors.border}` }}
        >
          <Link
            href="/dashboard"
            className={cn(
              "flex items-center gap-2 transition-all duration-300",
              showExpanded ? "" : "justify-center"
            )}
          >
            <SpikeLogo size={36} className="flex-shrink-0 transition-transform duration-300 hover:scale-105" />
            <span
              className={cn(
                "font-serif text-xl font-semibold tracking-tight transition-all duration-300 whitespace-nowrap",
                showExpanded ? "opacity-100 w-auto" : "opacity-0 w-0 overflow-hidden"
              )}
              style={{ color: colors.accent }}
            >
              spike
            </span>
          </Link>

          {/* Collapse Toggle */}
          {showExpanded && (
            <button
              onClick={() => setIsCollapsed(!isCollapsed)}
              className="p-1.5 rounded-lg transition-all duration-300 flex-shrink-0"
              style={{ color: colors.textMuted }}
              onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = colors.bgHover)}
              onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
            >
              {isCollapsed ? (
                <ChevronRight className="w-4 h-4" />
              ) : (
                <ChevronLeft className="w-4 h-4" />
              )}
            </button>
          )}
        </div>

        {/* Live Portfolio Widget */}
        <div
          className={cn(
            "mx-3 mt-4 rounded-xl transition-all duration-300 overflow-hidden",
            showExpanded ? "p-3 opacity-100" : "p-0 opacity-0 h-0 mt-0"
          )}
          style={{
            backgroundColor: portfolioData.isPositive ? colors.gainBg : colors.bgHover,
            border: showExpanded ? `1px solid ${portfolioData.isPositive ? `${colors.gain}30` : colors.border}` : "none",
          }}
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-[10px] uppercase tracking-wider" style={{ color: colors.textMuted }}>
                Portfolio
              </p>
              <p className="text-lg font-bold" style={{ color: colors.textPrimary }}>
                {portfolioData.value}
              </p>
            </div>
            <div className="flex items-center gap-1">
              {portfolioData.isPositive ? (
                <TrendingUp className="w-4 h-4" style={{ color: colors.gain }} />
              ) : (
                <TrendingDown className="w-4 h-4" style={{ color: colors.loss }} />
              )}
              <span
                className="text-sm font-semibold"
                style={{ color: portfolioData.isPositive ? colors.gain : colors.loss }}
              >
                {portfolioData.isPositive ? "+" : ""}{portfolioData.change}%
              </span>
            </div>
          </div>
        </div>

        {/* Collapsed Portfolio Indicator */}
        <div
          className={cn(
            "mx-auto mt-4 w-10 h-10 rounded-xl flex items-center justify-center transition-all duration-300",
            !showExpanded ? "opacity-100 scale-100" : "opacity-0 scale-75 h-0 mt-0"
          )}
          style={{ backgroundColor: colors.gainBg }}
        >
          <TrendingUp className="w-5 h-5" style={{ color: colors.gain }} />
        </div>

        {/* Navigation */}
        <nav className="flex-1 py-4 px-2 overflow-y-auto">
          <div className="space-y-1">
            {navigation.map((item, index) => {
              const isActive = pathname === item.href;
              return (
                <Link
                  key={item.name}
                  href={item.href}
                  className={cn(
                    "group relative flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all duration-200",
                    showExpanded ? "justify-start" : "justify-center"
                  )}
                  style={{
                    backgroundColor: isActive ? colors.accentLight : "transparent",
                    color: isActive ? colors.accent : colors.textSecondary,
                  }}
                  onMouseEnter={(e) => {
                    if (!isActive) e.currentTarget.style.backgroundColor = colors.bgHover;
                  }}
                  onMouseLeave={(e) => {
                    if (!isActive) e.currentTarget.style.backgroundColor = "transparent";
                  }}
                >
                  <item.icon
                    className="w-5 h-5 flex-shrink-0 transition-transform duration-200 group-hover:scale-110"
                  />
                  <span
                    className={cn(
                      "text-sm font-medium whitespace-nowrap transition-all duration-300",
                      showExpanded ? "opacity-100 translate-x-0" : "opacity-0 -translate-x-2 w-0 overflow-hidden"
                    )}
                  >
                    {item.name}
                  </span>

                  {/* Tooltip for collapsed state */}
                  {!showExpanded && (
                    <div
                      className="absolute left-full ml-2 px-2 py-1 rounded-md text-xs font-medium whitespace-nowrap opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity duration-200 z-50"
                      style={{
                        backgroundColor: colors.textPrimary,
                        color: "#FFFFFF",
                      }}
                    >
                      {item.name}
                    </div>
                  )}

                  {/* Active indicator bar */}
                  {isActive && (
                    <div
                      className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-6 rounded-r-full"
                      style={{ backgroundColor: colors.accent }}
                    />
                  )}
                </Link>
              );
            })}
          </div>

          {/* AI Tools Section */}
          <div className="mt-6">
            {showExpanded ? (
              <div className="px-3 mb-2 flex items-center gap-2">
                <div className="h-px flex-1" style={{ backgroundColor: colors.border }} />
                <span
                  className="text-[10px] font-semibold uppercase tracking-wider flex items-center gap-1"
                  style={{ color: colors.textMuted }}
                >
                  <Sparkles className="w-3 h-3" />
                  AI Tools
                </span>
                <div className="h-px flex-1" style={{ backgroundColor: colors.border }} />
              </div>
            ) : (
              <div className="mx-auto w-8 h-px my-3" style={{ backgroundColor: colors.border }} />
            )}

            <div className="space-y-1">
              {aiTools.map((item) => {
                const isActive = pathname === item.href;
                return (
                  <Link
                    key={item.name}
                    href={item.href}
                    className={cn(
                      "group relative flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all duration-200",
                      showExpanded ? "justify-start" : "justify-center"
                    )}
                    style={{
                      backgroundColor: isActive ? colors.accentLight : "transparent",
                      color: isActive ? colors.accent : colors.textSecondary,
                    }}
                    onMouseEnter={(e) => {
                      if (!isActive) e.currentTarget.style.backgroundColor = colors.bgHover;
                    }}
                    onMouseLeave={(e) => {
                      if (!isActive) e.currentTarget.style.backgroundColor = "transparent";
                    }}
                  >
                    <item.icon className="w-5 h-5 flex-shrink-0 transition-transform duration-200 group-hover:scale-110" />
                    <span
                      className={cn(
                        "text-sm font-medium whitespace-nowrap transition-all duration-300 flex-1",
                        showExpanded ? "opacity-100 translate-x-0" : "opacity-0 -translate-x-2 w-0 overflow-hidden"
                      )}
                    >
                      {item.name}
                    </span>
                    {item.badge && showExpanded && (
                      <span
                        className="px-1.5 py-0.5 text-[10px] font-bold rounded"
                        style={{
                          background: `linear-gradient(135deg, #7C3AED, ${colors.accent})`,
                          color: "#FFFFFF",
                        }}
                      >
                        {item.badge}
                      </span>
                    )}

                    {/* Pro badge for collapsed */}
                    {item.badge && !showExpanded && (
                      <div
                        className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 rounded-full border-2 border-white"
                        style={{ background: `linear-gradient(135deg, #7C3AED, ${colors.accent})` }}
                      />
                    )}

                    {/* Tooltip for collapsed state */}
                    {!showExpanded && (
                      <div
                        className="absolute left-full ml-2 px-2 py-1 rounded-md text-xs font-medium whitespace-nowrap opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity duration-200 z-50"
                        style={{
                          backgroundColor: colors.textPrimary,
                          color: "#FFFFFF",
                        }}
                      >
                        {item.name}
                        {item.badge && (
                          <span className="ml-1 text-[10px] opacity-75">({item.badge})</span>
                        )}
                      </div>
                    )}

                    {/* Active indicator bar */}
                    {isActive && (
                      <div
                        className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-6 rounded-r-full"
                        style={{ backgroundColor: colors.accent }}
                      />
                    )}
                  </Link>
                );
              })}
            </div>
          </div>
        </nav>

        {/* Market Status */}
        <div
          className={cn(
            "mx-3 mb-3 rounded-xl transition-all duration-300 overflow-hidden",
            showExpanded ? "p-3 opacity-100" : "p-0 opacity-0 h-0 mb-0"
          )}
          style={{
            backgroundColor: colors.bgHover,
            border: showExpanded ? `1px solid ${colors.border}` : "none",
          }}
        >
          <div className="flex items-center gap-2 mb-2">
            <div
              className="w-2 h-2 rounded-full animate-pulse"
              style={{ backgroundColor: marketStatus.isOpen ? colors.gain : colors.loss }}
            />
            <span className="text-[10px] uppercase tracking-wider" style={{ color: colors.textMuted }}>
              {marketStatus.isOpen ? "Market Open" : "Market Closed"}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xs" style={{ color: colors.textSecondary }}>NIFTY 50</span>
            <div className="flex items-center gap-1">
              <span className="text-xs font-semibold" style={{ color: colors.textPrimary }}>
                {marketStatus.nifty.value}
              </span>
              <span
                className="text-[10px] font-medium"
                style={{ color: marketStatus.nifty.change >= 0 ? colors.gain : colors.loss }}
              >
                +{marketStatus.nifty.change}%
              </span>
            </div>
          </div>
        </div>

        {/* Bottom Section */}
        <div className="p-2" style={{ borderTop: `1px solid ${colors.border}` }}>
          <Link
            href="/dashboard/settings"
            className={cn(
              "group relative flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all duration-200",
              showExpanded ? "justify-start" : "justify-center"
            )}
            style={{
              backgroundColor: pathname === "/dashboard/settings" ? colors.accentLight : "transparent",
              color: pathname === "/dashboard/settings" ? colors.accent : colors.textSecondary,
            }}
            onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = colors.bgHover)}
            onMouseLeave={(e) => {
              if (pathname !== "/dashboard/settings") {
                e.currentTarget.style.backgroundColor = "transparent";
              }
            }}
          >
            <Settings className="w-5 h-5 flex-shrink-0" />
            <span
              className={cn(
                "text-sm font-medium transition-all duration-300",
                showExpanded ? "opacity-100 translate-x-0" : "opacity-0 -translate-x-2 w-0 overflow-hidden"
              )}
            >
              Settings
            </span>

            {/* Tooltip for collapsed state */}
            {!showExpanded && (
              <div
                className="absolute left-full ml-2 px-2 py-1 rounded-md text-xs font-medium whitespace-nowrap opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity duration-200 z-50"
                style={{
                  backgroundColor: colors.textPrimary,
                  color: "#FFFFFF",
                }}
              >
                Settings
              </div>
            )}
          </Link>

          {/* Keyboard shortcut hint */}
          <div
            className={cn(
              "mt-2 px-3 py-2 flex items-center justify-center gap-1 transition-all duration-300",
              showExpanded ? "opacity-100" : "opacity-0 h-0 mt-0 overflow-hidden"
            )}
          >
            <kbd
              className="px-1.5 py-0.5 text-[10px] rounded"
              style={{ backgroundColor: colors.bgHover, color: colors.textMuted }}
            >
              ⌘
            </kbd>
            <kbd
              className="px-1.5 py-0.5 text-[10px] rounded"
              style={{ backgroundColor: colors.bgHover, color: colors.textMuted }}
            >
              [
            </kbd>
            <span className="text-[10px]" style={{ color: colors.textMuted }}>
              to {isCollapsed ? "expand" : "collapse"}
            </span>
          </div>
        </div>
      </aside>

      {/* Floating expand button when fully collapsed and not hovering */}
      {isCollapsed && !isHovering && (
        <button
          onClick={() => setIsCollapsed(false)}
          className="fixed left-[76px] top-1/2 -translate-y-1/2 z-50 p-1.5 rounded-r-lg shadow-lg transition-all duration-300 hover:pl-3 hidden lg:block"
          style={{
            backgroundColor: colors.bg,
            border: `1px solid ${colors.border}`,
            borderLeft: "none",
            color: colors.textMuted,
          }}
        >
          <ChevronRight className="w-4 h-4" />
        </button>
      )}
    </>
  );
}
