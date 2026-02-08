import { Suspense } from "react";
import { PortfolioSummary } from "@/components/dashboard/portfolio-summary";
import { MarketOverview } from "@/components/dashboard/market-overview";
import { TopMovers } from "@/components/dashboard/top-movers";
import { WatchlistWidget } from "@/components/dashboard/watchlist-widget";
import { AIInsights } from "@/components/dashboard/ai-insights";
import { QuickActions } from "@/components/dashboard/quick-actions";

// Color constants
const colors = {
  textPrimary: "#0D3331",
  textMuted: "#6B9B94",
};

export default function DashboardPage() {
  return (
    <div className="space-y-6">
      {/* Welcome Section */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold" style={{ color: colors.textPrimary }}>Good Morning!</h1>
          <p style={{ color: colors.textMuted }}>
            Here's what's happening with your investments today.
          </p>
        </div>
        <QuickActions />
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column - Portfolio & Market */}
        <div className="lg:col-span-2 space-y-6">
          <Suspense fallback={<WidgetSkeleton className="h-48" />}>
            <PortfolioSummary />
          </Suspense>

          <Suspense fallback={<WidgetSkeleton className="h-64" />}>
            <MarketOverview />
          </Suspense>

          <Suspense fallback={<WidgetSkeleton className="h-80" />}>
            <TopMovers />
          </Suspense>
        </div>

        {/* Right Column - Watchlist & AI */}
        <div className="space-y-6">
          <Suspense fallback={<WidgetSkeleton className="h-96" />}>
            <WatchlistWidget />
          </Suspense>

          <Suspense fallback={<WidgetSkeleton className="h-64" />}>
            <AIInsights />
          </Suspense>
        </div>
      </div>
    </div>
  );
}

function WidgetSkeleton({ className }: { className?: string }) {
  return (
    <div
      className={`rounded-2xl animate-pulse ${className}`}
      style={{ backgroundColor: "#FFFFFF", border: "1px solid #B8DDD7" }}
    />
  );
}
