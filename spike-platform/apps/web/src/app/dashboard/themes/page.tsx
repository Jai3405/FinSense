"use client";

import { useState } from "react";
import {
  Zap,
  Cpu,
  Leaf,
  Building2,
  Smartphone,
  Heart,
  Factory,
  Globe,
  TrendingUp,
  TrendingDown,
  ChevronRight,
  Star,
  Info,
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
};

interface Theme {
  id: string;
  name: string;
  description: string;
  icon: React.ElementType;
  iconBg: string;
  iconColor: string;
  performance1M: number;
  performance1Y: number;
  stockCount: number;
  topStocks: { symbol: string; change: number }[];
  marketCap: string;
  trending: boolean;
}

const themes: Theme[] = [
  {
    id: "ev",
    name: "Electric Vehicles",
    description: "EV manufacturers, battery tech, and charging infrastructure",
    icon: Zap,
    iconBg: "#FFF3E0",
    iconColor: "#F57C00",
    performance1M: 8.5,
    performance1Y: 45.2,
    stockCount: 24,
    topStocks: [
      { symbol: "TATAMOTORS", change: 2.4 },
      { symbol: "M&M", change: 1.8 },
      { symbol: "EXIDEIND", change: -0.5 },
    ],
    marketCap: "₹12.5L Cr",
    trending: true,
  },
  {
    id: "ai",
    name: "AI & Technology",
    description: "Artificial intelligence, cloud computing, and IT services",
    icon: Cpu,
    iconBg: "#E3F2FD",
    iconColor: "#1976D2",
    performance1M: 5.2,
    performance1Y: 32.8,
    stockCount: 35,
    topStocks: [
      { symbol: "TCS", change: 1.2 },
      { symbol: "INFY", change: 0.8 },
      { symbol: "WIPRO", change: -0.3 },
    ],
    marketCap: "₹28.4L Cr",
    trending: true,
  },
  {
    id: "green",
    name: "Green Energy",
    description: "Renewable energy, solar, wind, and sustainable solutions",
    icon: Leaf,
    iconBg: "#E8F5E9",
    iconColor: "#388E3C",
    performance1M: 12.3,
    performance1Y: 58.6,
    stockCount: 18,
    topStocks: [
      { symbol: "ADANIGREEN", change: 3.5 },
      { symbol: "TATAPOWER", change: 2.1 },
      { symbol: "NTPC", change: 1.4 },
    ],
    marketCap: "₹8.2L Cr",
    trending: true,
  },
  {
    id: "infra",
    name: "Infrastructure",
    description: "Construction, cement, roads, and urban development",
    icon: Building2,
    iconBg: "#FBE9E7",
    iconColor: "#D84315",
    performance1M: 3.8,
    performance1Y: 28.4,
    stockCount: 42,
    topStocks: [
      { symbol: "LT", change: 1.5 },
      { symbol: "ULTRACEMCO", change: 0.9 },
      { symbol: "ADANIENT", change: 2.2 },
    ],
    marketCap: "₹15.8L Cr",
    trending: false,
  },
  {
    id: "digital",
    name: "Digital India",
    description: "Fintech, e-commerce, digital payments, and platforms",
    icon: Smartphone,
    iconBg: "#F3E5F5",
    iconColor: "#7B1FA2",
    performance1M: 6.7,
    performance1Y: 41.2,
    stockCount: 28,
    topStocks: [
      { symbol: "PAYTM", change: -1.2 },
      { symbol: "ZOMATO", change: 4.5 },
      { symbol: "NYKAA", change: 1.8 },
    ],
    marketCap: "₹6.4L Cr",
    trending: true,
  },
  {
    id: "pharma",
    name: "Healthcare & Pharma",
    description: "Pharmaceuticals, hospitals, and healthcare services",
    icon: Heart,
    iconBg: "#FCE4EC",
    iconColor: "#C2185B",
    performance1M: 2.1,
    performance1Y: 18.5,
    stockCount: 32,
    topStocks: [
      { symbol: "SUNPHARMA", change: 0.8 },
      { symbol: "DRREDDY", change: 1.2 },
      { symbol: "APOLLOHOSP", change: 0.5 },
    ],
    marketCap: "₹10.2L Cr",
    trending: false,
  },
  {
    id: "manufacturing",
    name: "Make in India",
    description: "Manufacturing, defense, and industrial production",
    icon: Factory,
    iconBg: "#ECEFF1",
    iconColor: "#546E7A",
    performance1M: 4.5,
    performance1Y: 35.2,
    stockCount: 38,
    topStocks: [
      { symbol: "HAL", change: 2.8 },
      { symbol: "BEL", change: 1.9 },
      { symbol: "BHEL", change: 0.6 },
    ],
    marketCap: "₹9.8L Cr",
    trending: false,
  },
  {
    id: "global",
    name: "Export Champions",
    description: "Companies with significant global revenue exposure",
    icon: Globe,
    iconBg: "#E0F7FA",
    iconColor: "#0097A7",
    performance1M: 3.2,
    performance1Y: 22.8,
    stockCount: 25,
    topStocks: [
      { symbol: "TCS", change: 1.2 },
      { symbol: "INFY", change: 0.8 },
      { symbol: "TECHM", change: -0.4 },
    ],
    marketCap: "₹22.1L Cr",
    trending: false,
  },
];

export default function ThemesPage() {
  const [selectedTheme, setSelectedTheme] = useState<string | null>(null);

  return (
    <div className="max-w-6xl">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-xl font-semibold mb-1" style={{ color: colors.textPrimary }}>
          Investment Themes
        </h1>
        <p className="text-sm" style={{ color: colors.textMuted }}>
          Discover curated investment themes aligned with India's growth story
        </p>
      </div>

      {/* Trending Themes */}
      <div className="mb-8">
        <div className="flex items-center gap-2 mb-4">
          <TrendingUp className="w-4 h-4" style={{ color: colors.gain }} />
          <h2 className="text-sm font-semibold" style={{ color: colors.textPrimary }}>
            Trending Themes
          </h2>
        </div>
        <div className="grid grid-cols-4 gap-4">
          {themes
            .filter((t) => t.trending)
            .map((theme) => {
              const Icon = theme.icon;
              const isPositive = theme.performance1M >= 0;
              return (
                <div
                  key={theme.id}
                  className="rounded-xl p-4 cursor-pointer transition-all"
                  style={{ backgroundColor: colors.bg, border: `1px solid ${colors.border}` }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor = colors.bgHover;
                    e.currentTarget.style.transform = "translateY(-2px)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = colors.bg;
                    e.currentTarget.style.transform = "translateY(0)";
                  }}
                >
                  <div className="flex items-start justify-between mb-3">
                    <div
                      className="w-10 h-10 rounded-lg flex items-center justify-center"
                      style={{ backgroundColor: theme.iconBg }}
                    >
                      <Icon className="w-5 h-5" style={{ color: theme.iconColor }} />
                    </div>
                    <Star className="w-4 h-4" style={{ color: "#F5A623" }} fill="#F5A623" />
                  </div>
                  <h3 className="text-sm font-semibold mb-1" style={{ color: colors.textPrimary }}>
                    {theme.name}
                  </h3>
                  <p className="text-xs mb-3 line-clamp-2" style={{ color: colors.textMuted }}>
                    {theme.description}
                  </p>
                  <div className="flex items-center justify-between">
                    <span className="text-xs" style={{ color: colors.textMuted }}>
                      {theme.stockCount} stocks
                    </span>
                    <span
                      className="text-sm font-semibold"
                      style={{ color: isPositive ? colors.gain : colors.loss }}
                    >
                      {isPositive ? "+" : ""}{theme.performance1M.toFixed(1)}%
                    </span>
                  </div>
                </div>
              );
            })}
        </div>
      </div>

      {/* All Themes */}
      <div
        className="rounded-xl overflow-hidden"
        style={{ backgroundColor: colors.bg, border: `1px solid ${colors.border}` }}
      >
        <div className="px-4 py-3" style={{ borderBottom: `1px solid ${colors.border}` }}>
          <h2 className="text-sm font-semibold" style={{ color: colors.textPrimary }}>
            All Themes
          </h2>
        </div>
        <div>
          {themes.map((theme, index) => {
            const Icon = theme.icon;
            const isPositive1M = theme.performance1M >= 0;
            const isPositive1Y = theme.performance1Y >= 0;
            return (
              <div
                key={theme.id}
                className="px-4 py-4 cursor-pointer transition-colors"
                style={{
                  borderBottom: index < themes.length - 1 ? `1px solid ${colors.border}` : "none",
                }}
                onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = colors.bgHover)}
                onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div
                      className="w-12 h-12 rounded-xl flex items-center justify-center"
                      style={{ backgroundColor: theme.iconBg }}
                    >
                      <Icon className="w-6 h-6" style={{ color: theme.iconColor }} />
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <h3 className="text-sm font-semibold" style={{ color: colors.textPrimary }}>
                          {theme.name}
                        </h3>
                        {theme.trending && (
                          <span
                            className="px-1.5 py-0.5 text-[10px] font-medium rounded"
                            style={{ backgroundColor: colors.gainBg, color: colors.gain }}
                          >
                            Trending
                          </span>
                        )}
                      </div>
                      <p className="text-xs mt-0.5" style={{ color: colors.textMuted }}>
                        {theme.stockCount} stocks · {theme.marketCap}
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center gap-8">
                    {/* Top Stocks */}
                    <div className="hidden lg:flex items-center gap-3">
                      {theme.topStocks.map((stock) => (
                        <div
                          key={stock.symbol}
                          className="px-2 py-1 rounded text-xs"
                          style={{ backgroundColor: colors.bgMint }}
                        >
                          <span style={{ color: colors.textSecondary }}>{stock.symbol}</span>
                          <span
                            className="ml-1 font-medium"
                            style={{ color: stock.change >= 0 ? colors.gain : colors.loss }}
                          >
                            {stock.change >= 0 ? "+" : ""}{stock.change.toFixed(1)}%
                          </span>
                        </div>
                      ))}
                    </div>

                    {/* Performance */}
                    <div className="flex items-center gap-6">
                      <div className="text-right">
                        <p className="text-xs" style={{ color: colors.textMuted }}>1M</p>
                        <p
                          className="text-sm font-semibold"
                          style={{ color: isPositive1M ? colors.gain : colors.loss }}
                        >
                          {isPositive1M ? "+" : ""}{theme.performance1M.toFixed(1)}%
                        </p>
                      </div>
                      <div className="text-right">
                        <p className="text-xs" style={{ color: colors.textMuted }}>1Y</p>
                        <p
                          className="text-sm font-semibold"
                          style={{ color: isPositive1Y ? colors.gain : colors.loss }}
                        >
                          {isPositive1Y ? "+" : ""}{theme.performance1Y.toFixed(1)}%
                        </p>
                      </div>
                    </div>

                    <ChevronRight className="w-5 h-5" style={{ color: colors.border }} />
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Info Banner */}
      <div
        className="mt-6 rounded-xl p-4"
        style={{ backgroundColor: colors.bgMint, border: `1px solid ${colors.border}` }}
      >
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 flex-shrink-0 mt-0.5" style={{ color: colors.textMuted }} />
          <div>
            <p className="text-sm" style={{ color: colors.textSecondary }}>
              Investment themes are curated baskets of stocks aligned with long-term structural trends.
              Each theme is powered by SPIKE's AI which continuously monitors and rebalances based on
              fundamentals and market conditions.
            </p>
            <button className="text-sm mt-2 font-medium" style={{ color: colors.accent }}>
              Learn how themes work
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
