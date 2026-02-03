"use client";

import { useUser, UserButton } from "@clerk/nextjs";
import { Bell, Menu, Search, TrendingUp, TrendingDown } from "lucide-react";
import { useEffect, useState } from "react";

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
    { name: "NIFTY BANK", value: 46850.0, change: -125.75, changePercent: -0.27 },
  ]);
  const [isMarketOpen, setIsMarketOpen] = useState(false);

  useEffect(() => {
    // Check if market is open (9:15 AM - 3:30 PM IST, Mon-Fri)
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
    <header className="sticky top-0 z-40 border-b border-white/10 bg-slate-950/80 backdrop-blur-xl">
      <div className="flex h-16 items-center justify-between px-6">
        {/* Mobile menu button */}
        <button className="lg:hidden p-2 rounded-lg hover:bg-white/5">
          <Menu className="h-6 w-6 text-slate-400" />
        </button>

        {/* Market Indices */}
        <div className="hidden md:flex items-center gap-6">
          <div className="flex items-center gap-2">
            <span
              className={`w-2 h-2 rounded-full ${
                isMarketOpen ? "bg-green-500 animate-pulse" : "bg-red-500"
              }`}
            />
            <span className="text-xs text-slate-400">
              {isMarketOpen ? "Market Open" : "Market Closed"}
            </span>
          </div>
          <div className="flex items-center gap-4">
            {indices.map((index) => (
              <div key={index.name} className="flex items-center gap-2">
                <span className="text-sm text-slate-400">{index.name}</span>
                <span className="text-sm font-medium text-white">
                  {index.value.toLocaleString("en-IN")}
                </span>
                <span
                  className={`flex items-center gap-0.5 text-xs font-medium ${
                    index.change >= 0 ? "text-spike-bull" : "text-spike-bear"
                  }`}
                >
                  {index.change >= 0 ? (
                    <TrendingUp className="w-3 h-3" />
                  ) : (
                    <TrendingDown className="w-3 h-3" />
                  )}
                  {index.changePercent >= 0 ? "+" : ""}
                  {index.changePercent.toFixed(2)}%
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Search */}
        <div className="flex-1 max-w-md mx-8 hidden lg:block">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
            <input
              type="text"
              placeholder="Search stocks, themes, or ask AI..."
              className="w-full h-10 pl-10 pr-4 rounded-xl bg-white/5 border border-white/10 text-white placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-spike-primary/50 focus:border-spike-primary"
            />
            <kbd className="absolute right-3 top-1/2 -translate-y-1/2 px-2 py-0.5 text-xs text-slate-400 bg-white/5 rounded">
              ⌘K
            </kbd>
          </div>
        </div>

        {/* Right section */}
        <div className="flex items-center gap-4">
          {/* Notifications */}
          <button className="relative p-2 rounded-lg hover:bg-white/5 transition">
            <Bell className="h-5 w-5 text-slate-400" />
            <span className="absolute top-1 right-1 w-2 h-2 bg-spike-primary rounded-full" />
          </button>

          {/* User */}
          <div className="flex items-center gap-3">
            <div className="hidden sm:block text-right">
              <div className="text-sm font-medium text-white">
                {user?.firstName || "Investor"}
              </div>
              <div className="text-xs text-slate-400">Pro Plan</div>
            </div>
            <UserButton
              afterSignOutUrl="/"
              appearance={{
                elements: {
                  avatarBox: "w-10 h-10",
                },
              }}
            />
          </div>
        </div>
      </div>
    </header>
  );
}
