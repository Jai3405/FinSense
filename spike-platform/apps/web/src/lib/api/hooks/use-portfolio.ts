"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@clerk/nextjs";
import { apiClient } from "../client";
import type { PortfolioSummary, Holding, AddHoldingRequest } from "../types";

export function usePortfolioSummary() {
  const { getToken } = useAuth();

  return useQuery({
    queryKey: ["portfolio", "summary"],
    queryFn: async () => {
      const token = await getToken();
      return apiClient<PortfolioSummary>("/portfolio/summary", {
        token: token || undefined,
      });
    },
  });
}

export function useHoldings() {
  const { getToken } = useAuth();

  return useQuery({
    queryKey: ["portfolio", "holdings"],
    queryFn: async () => {
      const token = await getToken();
      return apiClient<Holding[]>("/portfolio/holdings", {
        token: token || undefined,
      });
    },
  });
}

export function useAddHolding() {
  const { getToken } = useAuth();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (data: AddHoldingRequest) => {
      const token = await getToken();
      return apiClient<{ status: string; holding: Holding }>(
        "/portfolio/holdings",
        {
          method: "POST",
          body: JSON.stringify(data),
          token: token || undefined,
        }
      );
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["portfolio"] });
    },
  });
}

export function useRemoveHolding() {
  const { getToken } = useAuth();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (symbol: string) => {
      const token = await getToken();
      return apiClient<{ status: string; symbol: string }>(
        `/portfolio/holdings/${symbol}`,
        {
          method: "DELETE",
          token: token || undefined,
        }
      );
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["portfolio"] });
    },
  });
}
