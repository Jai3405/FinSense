"use client";

import { useEffect, useRef, useState } from "react";
import { getPriceWebSocket, type LiveQuote } from "../websocket";

export function useLivePrices(symbols: string[]) {
  const [prices, setPrices] = useState<Record<string, LiveQuote>>({});
  const wsRef = useRef(getPriceWebSocket());

  useEffect(() => {
    const ws = wsRef.current;
    ws.connect();

    if (symbols.length > 0) {
      ws.subscribe(symbols);
    }

    const unsubscribe = ws.onQuote((symbol, quote) => {
      setPrices((prev) => ({ ...prev, [symbol]: quote }));
    });

    return () => {
      unsubscribe();
      if (symbols.length > 0) {
        ws.unsubscribe(symbols);
      }
    };
  }, [JSON.stringify(symbols)]); // eslint-disable-line react-hooks/exhaustive-deps

  return prices;
}
