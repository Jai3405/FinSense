"use client";

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";

export interface LiveQuote {
  price: number;
  change: number;
  change_percent: number;
  volume: number;
  high: number;
  low: number;
  timestamp: string;
}

type QuoteCallback = (symbol: string, quote: LiveQuote) => void;

class PriceWebSocket {
  private ws: WebSocket | null = null;
  private callbacks: Set<QuoteCallback> = new Set();
  private subscribedSymbols: Set<string> = new Set();
  private reconnectTimer: NodeJS.Timeout | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;

  connect() {
    if (this.ws?.readyState === WebSocket.OPEN) return;

    try {
      this.ws = new WebSocket(`${WS_URL}/api/v1/ws/prices`);

      this.ws.onopen = () => {
        console.log("[WS] Connected");
        this.reconnectAttempts = 0;
        // Resubscribe to previously subscribed symbols
        if (this.subscribedSymbols.size > 0) {
          this.ws?.send(
            JSON.stringify({
              action: "subscribe",
              symbols: Array.from(this.subscribedSymbols),
            })
          );
        }
      };

      this.ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === "quote") {
          this.callbacks.forEach((cb) => cb(msg.symbol, msg.data));
        }
      };

      this.ws.onclose = () => {
        console.log("[WS] Disconnected");
        this.attemptReconnect();
      };

      this.ws.onerror = (error) => {
        console.error("[WS] Error:", error);
      };
    } catch (e) {
      console.error("[WS] Connection failed:", e);
      this.attemptReconnect();
    }
  }

  private attemptReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) return;
    if (this.reconnectTimer) return;

    this.reconnectAttempts++;
    const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);

    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, delay);
  }

  subscribe(symbols: string[]) {
    symbols.forEach((s) => this.subscribedSymbols.add(s));
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ action: "subscribe", symbols }));
    }
  }

  unsubscribe(symbols: string[]) {
    symbols.forEach((s) => this.subscribedSymbols.delete(s));
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ action: "unsubscribe", symbols }));
    }
  }

  onQuote(callback: QuoteCallback): () => void {
    this.callbacks.add(callback);
    return () => this.callbacks.delete(callback);
  }

  disconnect() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.ws?.close();
    this.ws = null;
  }
}

// Singleton
let instance: PriceWebSocket | null = null;

export function getPriceWebSocket(): PriceWebSocket {
  if (!instance) {
    instance = new PriceWebSocket();
  }
  return instance;
}
