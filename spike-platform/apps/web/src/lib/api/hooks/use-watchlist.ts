"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@clerk/nextjs";
import { apiClient } from "../client";
import type { WatchlistStock, AddToWatchlistRequest } from "../types";

export function useWatchlist() {
  const { getToken } = useAuth();

  return useQuery({
    queryKey: ["watchlist"],
    queryFn: async () => {
      const token = await getToken();
      return apiClient<WatchlistStock[]>("/watchlist/", {
        token: token || undefined,
      });
    },
  });
}

export function useAddToWatchlist() {
  const { getToken } = useAuth();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (data: AddToWatchlistRequest) => {
      const token = await getToken();
      return apiClient<{ status: string; symbol: string }>("/watchlist/", {
        method: "POST",
        body: JSON.stringify(data),
        token: token || undefined,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["watchlist"] });
    },
  });
}

export function useRemoveFromWatchlist() {
  const { getToken } = useAuth();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (symbol: string) => {
      const token = await getToken();
      return apiClient<{ status: string; symbol: string }>(
        `/watchlist/${symbol}`,
        {
          method: "DELETE",
          token: token || undefined,
        }
      );
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["watchlist"] });
    },
  });
}
