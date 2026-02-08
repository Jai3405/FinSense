"use client";

import { useUser, UserButton } from "@clerk/nextjs";
import { Bell, Menu, Search, TrendingUp, TrendingDown } from "lucide-react";
import { useEffect, useState } from "react";

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
        {/* Mobile menu */}
        <button
          className="lg:hidden p-2 -ml-2 rounded-lg transition-colors"
          style={{ color: colors.textSecondary }}
          onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = colors.bgHover)}
          onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
        >
          <Menu className="h-5 w-5" />
        </button>

        {/* Market Indices */}
        <div className="hidden md:flex items-center gap-6">
          <div className="flex items-center gap-1.5">
            <span
              className="w-1.5 h-1.5 rounded-full"
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

        {/* Search */}
        <div className="flex-1 max-w-md mx-6 hidden lg:block">
          <div className="relative">
            <Search
              className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4"
              style={{ color: colors.textMuted }}
            />
            <input
              type="text"
              placeholder="Search stocks, ETFs, or mutual funds"
              className="w-full h-9 pl-9 pr-3 text-sm rounded-lg transition-colors focus:outline-none"
              style={{
                backgroundColor: colors.inputBg,
                border: `1px solid ${colors.border}`,
                color: colors.textPrimary,
              }}
              onFocus={(e) => {
                e.currentTarget.style.backgroundColor = "#FFFFFF";
                e.currentTarget.style.borderColor = colors.accent;
              }}
              onBlur={(e) => {
                e.currentTarget.style.backgroundColor = colors.inputBg;
                e.currentTarget.style.borderColor = colors.border;
              }}
            />
          </div>
        </div>

        {/* Right section */}
        <div className="flex items-center gap-3">
          {/* Notifications */}
          <button
            className="relative p-2 rounded-lg transition-colors"
            style={{ color: colors.textSecondary }}
            onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = colors.bgHover)}
            onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
          >
            <Bell className="h-5 w-5" />
            <span
              className="absolute top-1.5 right-1.5 w-1.5 h-1.5 rounded-full"
              style={{ backgroundColor: colors.accent }}
            />
          </button>

          {/* User */}
          <div
            className="flex items-center gap-2 pl-3"
            style={{ borderLeft: `1px solid ${colors.border}` }}
          >
            <UserButton
              afterSignOutUrl="/"
              appearance={{
                elements: {
                  avatarBox: "w-8 h-8",
                },
              }}
            />
          </div>
        </div>
      </div>
    </header>
  );
}
