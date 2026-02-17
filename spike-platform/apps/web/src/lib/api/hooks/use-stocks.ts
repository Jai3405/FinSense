"use client";

import { useQuery } from "@tanstack/react-query";
import { useAuth } from "@clerk/nextjs";
import { apiClient } from "../client";
import type {
  StockQuote,
  StockSearchResult,
  StockInfo,
  StockHistory,
  TrendingStock,
} from "../types";

export function useStockQuote(symbol: string) {
  return useQuery({
    queryKey: ["stocks", "quote", symbol],
    queryFn: () => apiClient<StockQuote>(`/stocks/${symbol}/quote`),
    enabled: !!symbol,
    refetchInterval: 5000,
  });
}

export function useStockSearch(query: string) {
  return useQuery({
    queryKey: ["stocks", "search", query],
    queryFn: () =>
      apiClient<StockSearchResult[]>(
        `/stocks/search?q=${encodeURIComponent(query)}&limit=10`
      ),
    enabled: query.length >= 2,
  });
}

export function useStockInfo(symbol: string) {
  const { getToken } = useAuth();

  return useQuery({
    queryKey: ["stocks", "info", symbol],
    queryFn: async () => {
      const token = await getToken();
      return apiClient<StockInfo>(`/stocks/${symbol}/info`, {
        token: token || undefined,
      });
    },
    enabled: !!symbol,
  });
}

export function useStockHistory(
  symbol: string,
  period: string = "1M",
  interval: string = "1d"
) {
  return useQuery({
    queryKey: ["stocks", "history", symbol, period, interval],
    queryFn: () =>
      apiClient<StockHistory>(
        `/stocks/${symbol}/history?period=${period}&interval=${interval}`
      ),
    enabled: !!symbol,
  });
}

export function useTrendingStocks(limit: number = 10) {
  return useQuery({
    queryKey: ["stocks", "trending", limit],
    queryFn: () =>
      apiClient<TrendingStock[]>(`/stocks/trending?limit=${limit}`),
    refetchInterval: 30000,
  });
}
