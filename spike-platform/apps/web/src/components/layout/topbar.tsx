"use client";

import { useUser, UserButton } from "@clerk/nextjs";
import { Bell, Menu, Search, Command, PanelLeftClose, PanelLeft } from "lucide-react";
import { useEffect, useState } from "react";
import { useSidebar } from "@/components/providers/sidebar-provider";

// Color constants
const colors = {
  bg: "#FFFFFF",
  bgHover: "#F0FBF9",
  border: "#B8DDD7",
  textPrimary: "#0D3331",
  textSecondary: "#3D6B66",
  textMuted: "#6B9B94",
  accent: "#00897B",
  gain: "#00B386",
  loss: "#F45B69",
  inputBg: "#F8FFFE",
};

interface MarketIndex {
  name: string;
  value: number;
  change: number;
  changePercent: number;
}

export function TopBar() {
  const { user } = useUser();
  const { isCollapsed, setIsCollapsed } = useSidebar();
  const [indices, setIndices] = useState<MarketIndex[]>([
    { name: "NIFTY 50", value: 22150.5, change: 125.3, changePercent: 0.57 },
    { name: "SENSEX", value: 72890.25, change: 380.5, changePercent: 0.52 },
    { name: "BANK NIFTY", value: 46850.0, change: -125.75, changePercent: -0.27 },
  ]);
  const [isMarketOpen, setIsMarketOpen] = useState(false);

  useEffect(() => {
    const checkMarketHours = () => {
      const now = new Date();
      const istOffset = 5.5 * 60 * 60 * 1000;
      const istTime = new Date(now.getTime() + istOffset);
      const hours = istTime.getUTCHours();
      const minutes = istTime.getUTCMinutes();
      const day = istTime.getUTCDay();

      const marketOpen = hours * 60 + minutes >= 9 * 60 + 15;
      const marketClose = hours * 60 + minutes <= 15 * 60 + 30;
      const isWeekday = day >= 1 && day <= 5;

      setIsMarketOpen(marketOpen && marketClose && isWeekday);
    };

    checkMarketHours();
    const interval = setInterval(checkMarketHours, 60000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header
      className="sticky top-0 z-40 h-14"
      style={{ backgroundColor: colors.bg, borderBottom: `1px solid ${colors.border}` }}
    >
      <div className="flex h-full items-center justify-between px-4">
        {/* Left section */}
        <div className="flex items-center gap-3">
          {/* Mobile menu */}
          <button
            className="lg:hidden p-2 -ml-2 rounded-lg transition-colors"
            style={{ color: colors.textSecondary }}
            onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = colors.bgHover)}
            onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
          >
            <Menu className="h-5 w-5" />
          </button>

          {/* Desktop sidebar toggle */}
          <button
            onClick={() => setIsCollapsed(!isCollapsed)}
            className="hidden lg:flex p-2 -ml-2 rounded-lg transition-colors items-center gap-2"
            style={{ color: colors.textSecondary }}
            onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = colors.bgHover)}
            onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
            title={isCollapsed ? "Expand sidebar (⌘[)" : "Collapse sidebar (⌘[)"}
          >
            {isCollapsed ? (
              <PanelLeft className="h-5 w-5" />
            ) : (
              <PanelLeftClose className="h-5 w-5" />
            )}
          </button>

          {/* Market Status & Indices */}
          <div className="hidden md:flex items-center gap-6 ml-2">
            <div className="flex items-center gap-1.5">
              <span
                className="w-1.5 h-1.5 rounded-full animate-pulse"
                style={{ backgroundColor: isMarketOpen ? colors.gain : colors.loss }}
              />
              <span className="text-xs" style={{ color: colors.textMuted }}>
                {isMarketOpen ? "Market Open" : "Closed"}
              </span>
            </div>

            <div className="flex items-center gap-5">
              {indices.map((index) => (
                <div key={index.name} className="flex items-center gap-2">
                  <span className="text-xs" style={{ color: colors.textMuted }}>
                    {index.name}
                  </span>
                  <span className="text-sm font-medium" style={{ color: colors.textPrimary }}>
                    {index.value.toLocaleString("en-IN")}
                  </span>
                  <span
                    className="text-xs font-medium"
                    style={{ color: index.change >= 0 ? colors.gain : colors.loss }}
                  >
                    {index.change >= 0 ? "+" : ""}
                    {index.changePercent.toFixed(2)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Search */}
        <div className="flex-1 max-w-md mx-6 hidden lg:block">
          <div className="relative group">
            <Search
              className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4"
              style={{ color: colors.textMuted }}
            />
            <input
              type="text"
              placeholder="Search stocks, ETFs, or mutual funds..."
              className="w-full h-9 pl-9 pr-16 text-sm rounded-xl transition-all duration-200 focus:outline-none"
              style={{
                backgroundColor: colors.inputBg,
                border: `1px solid ${colors.border}`,
                color: colors.textPrimary,
              }}
              onFocus={(e) => {
                e.currentTarget.style.backgroundColor = "#FFFFFF";
                e.currentTarget.style.borderColor = colors.accent;
                e.currentTarget.style.boxShadow = `0 0 0 3px ${colors.accent}15`;
              }}
              onBlur={(e) => {
                e.currentTarget.style.backgroundColor = colors.inputBg;
                e.currentTarget.style.borderColor = colors.border;
                e.currentTarget.style.boxShadow = "none";
              }}
            />
            {/* Keyboard shortcut hint */}
            <div
              className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-0.5 px-1.5 py-0.5 rounded-md"
              style={{ backgroundColor: colors.bgHover }}
            >
              <Command className="w-3 h-3" style={{ color: colors.textMuted }} />
              <span className="text-[10px] font-medium" style={{ color: colors.textMuted }}>K</span>
            </div>
          </div>
        </div>

        {/* Right section */}
        <div className="flex items-center gap-2">
          {/* Notifications */}
          <button
            className="relative p-2 rounded-xl transition-colors"
            style={{ color: colors.textSecondary }}
            onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = colors.bgHover)}
            onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
          >
            <Bell className="h-5 w-5" />
            <span
              className="absolute top-1.5 right-1.5 w-2 h-2 rounded-full border-2"
              style={{ backgroundColor: colors.accent, borderColor: colors.bg }}
            />
          </button>

          {/* User */}
          <div
            className="flex items-center gap-3 pl-3 ml-1"
            style={{ borderLeft: `1px solid ${colors.border}` }}
          >
            <UserButton
              afterSignOutUrl="/"
              appearance={{
                elements: {
                  avatarBox: "w-8 h-8 rounded-xl",
                },
              }}
            />
          </div>
        </div>
      </div>
    </header>
  );
}
