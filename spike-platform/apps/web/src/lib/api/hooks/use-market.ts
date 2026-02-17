"use client";

import { useQuery } from "@tanstack/react-query";
import { apiClient } from "../client";
import type {
  MarketIndex,
  SectorPerformance,
  MarketStatus,
  MarketRegime,
  MarketBreadth,
} from "../types";

export function useMarketIndices() {
  return useQuery({
    queryKey: ["market", "indices"],
    queryFn: () => apiClient<MarketIndex[]>("/market/indices"),
    refetchInterval: 10000,
  });
}

export function useSectorPerformance() {
  return useQuery({
    queryKey: ["market", "sectors"],
    queryFn: () => apiClient<SectorPerformance[]>("/market/sectors"),
    refetchInterval: 30000,
  });
}

export function useMarketStatus() {
  return useQuery({
    queryKey: ["market", "status"],
    queryFn: () => apiClient<MarketStatus>("/market/status"),
    refetchInterval: 60000,
  });
}

export function useMarketRegime() {
  return useQuery({
    queryKey: ["market", "regime"],
    queryFn: () => apiClient<MarketRegime>("/market/regime"),
    refetchInterval: 60000,
  });
}

export function useMarketBreadth() {
  return useQuery({
    queryKey: ["market", "breadth"],
    queryFn: () => apiClient<MarketBreadth>("/market/breadth"),
    refetchInterval: 30000,
  });
}
