"use client";

import { useState } from "react";
import {
  Sparkles,
  Send,
  Lightbulb,
  TrendingUp,
  Shield,
  Target,
  Clock,
  RotateCcw,
  Copy,
  ThumbsUp,
  ThumbsDown,
  ChevronRight,
  BarChart3,
  Zap,
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
  inputBg: "#F8FFFE",
};

const suggestedPrompts = [
  {
    icon: TrendingUp,
    title: "Build a momentum strategy",
    description: "Create a strategy based on price momentum and volume",
  },
  {
    icon: Shield,
    title: "Low volatility portfolio",
    description: "Design a defensive portfolio for uncertain markets",
  },
  {
    icon: Target,
    title: "Value investing approach",
    description: "Find undervalued stocks with strong fundamentals",
  },
  {
    icon: Zap,
    title: "Small cap growth picks",
    description: "Identify high-growth potential small cap stocks",
  },
];

const recentStrategies = [
  { name: "Dividend Aristocrats India", date: "2 days ago", performance: "+12.4%" },
  { name: "Tech Growth Portfolio", date: "1 week ago", performance: "+8.2%" },
  { name: "PSU Revival Basket", date: "2 weeks ago", performance: "+15.8%" },
];

interface Message {
  role: "user" | "assistant";
  content: string;
  stocks?: { symbol: string; allocation: number; reason: string }[];
}

const sampleConversation: Message[] = [
  {
    role: "user",
    content: "Create a balanced portfolio for long-term wealth creation with moderate risk",
  },
  {
    role: "assistant",
    content: "I've analyzed the current market conditions and your risk profile to create a diversified portfolio. Here's my recommendation:\n\n**Portfolio Strategy: Balanced Growth**\n\nThis portfolio combines large-cap stability with selective mid-cap exposure for growth. The allocation is designed to weather market volatility while capturing upside potential.",
    stocks: [
      { symbol: "HDFCBANK", allocation: 20, reason: "Strong retail franchise, consistent earnings growth" },
      { symbol: "TCS", allocation: 15, reason: "IT leader with robust order book and margins" },
      { symbol: "RELIANCE", allocation: 15, reason: "Diversified conglomerate with retail & telecom growth" },
      { symbol: "ICICIBANK", allocation: 12, reason: "Improving asset quality, digital banking leader" },
      { symbol: "INFY", allocation: 10, reason: "IT services with AI/cloud transformation" },
      { symbol: "BHARTIARTL", allocation: 10, reason: "Telecom duopoly beneficiary, ARPU growth" },
      { symbol: "TITAN", allocation: 10, reason: "Premium consumer play, strong brand" },
      { symbol: "SUNPHARMA", allocation: 8, reason: "Pharma leader with specialty portfolio" },
    ],
  },
];

export default function StrategyPage() {
  const [messages, setMessages] = useState<Message[]>(sampleConversation);
  const [inputValue, setInputValue] = useState("");
  const [isTyping, setIsTyping] = useState(false);

  const handleSend = () => {
    if (!inputValue.trim()) return;
    setMessages([...messages, { role: "user", content: inputValue }]);
    setInputValue("");
    setIsTyping(true);
    // Simulate AI response
    setTimeout(() => setIsTyping(false), 2000);
  };

  return (
    <div className="max-w-5xl mx-auto h-[calc(100vh-8rem)] flex flex-col">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center gap-3">
          <div
            className="w-10 h-10 rounded-xl flex items-center justify-center"
            style={{ background: `linear-gradient(135deg, ${colors.accent}, ${colors.gain})` }}
          >
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-semibold" style={{ color: colors.textPrimary }}>
              Strategy-GPT
            </h1>
            <p className="text-sm" style={{ color: colors.textMuted }}>
              AI-powered investment strategy builder
            </p>
          </div>
        </div>
      </div>

      <div className="flex-1 grid grid-cols-4 gap-6 min-h-0">
        {/* Sidebar */}
        <div className="space-y-4">
          {/* New Chat */}
          <button
            className="w-full flex items-center justify-center gap-2 py-2.5 rounded-xl text-sm font-medium text-white"
            style={{ background: `linear-gradient(135deg, ${colors.accent}, ${colors.gain})` }}
          >
            <Sparkles className="w-4 h-4" />
            New Strategy
          </button>

          {/* Recent Strategies */}
          <div
            className="rounded-xl p-4"
            style={{ backgroundColor: colors.bg, border: `1px solid ${colors.border}` }}
          >
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-xs font-semibold" style={{ color: colors.textPrimary }}>
                Recent Strategies
              </h3>
              <Clock className="w-3.5 h-3.5" style={{ color: colors.textMuted }} />
            </div>
            <div className="space-y-2">
              {recentStrategies.map((strategy) => (
                <button
                  key={strategy.name}
                  className="w-full text-left p-2.5 rounded-lg transition-colors"
                  style={{ backgroundColor: colors.bgMint }}
                  onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = colors.bgHover)}
                  onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = colors.bgMint)}
                >
                  <p className="text-xs font-medium truncate" style={{ color: colors.textPrimary }}>
                    {strategy.name}
                  </p>
                  <div className="flex items-center justify-between mt-1">
                    <span className="text-[10px]" style={{ color: colors.textMuted }}>
                      {strategy.date}
                    </span>
                    <span className="text-[10px] font-medium" style={{ color: colors.gain }}>
                      {strategy.performance}
                    </span>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Suggested Prompts */}
          <div
            className="rounded-xl p-4"
            style={{ backgroundColor: colors.bg, border: `1px solid ${colors.border}` }}
          >
            <div className="flex items-center gap-2 mb-3">
              <Lightbulb className="w-3.5 h-3.5" style={{ color: colors.accent }} />
              <h3 className="text-xs font-semibold" style={{ color: colors.textPrimary }}>
                Try asking
              </h3>
            </div>
            <div className="space-y-2">
              {suggestedPrompts.slice(0, 3).map((prompt) => {
                const Icon = prompt.icon;
                return (
                  <button
                    key={prompt.title}
                    className="w-full text-left p-2 rounded-lg transition-colors"
                    onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = colors.bgHover)}
                    onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
                    onClick={() => setInputValue(prompt.title)}
                  >
                    <p className="text-xs" style={{ color: colors.textSecondary }}>
                      {prompt.title}
                    </p>
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* Chat Area */}
        <div className="col-span-3 flex flex-col min-h-0">
          <div
            className="flex-1 rounded-xl p-4 overflow-y-auto"
            style={{ backgroundColor: colors.bg, border: `1px solid ${colors.border}` }}
          >
            {messages.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center text-center">
                <div
                  className="w-16 h-16 rounded-2xl flex items-center justify-center mb-4"
                  style={{ background: `linear-gradient(135deg, ${colors.accent}20, ${colors.gain}20)` }}
                >
                  <Sparkles className="w-8 h-8" style={{ color: colors.accent }} />
                </div>
                <h2 className="text-lg font-semibold mb-2" style={{ color: colors.textPrimary }}>
                  Start Building Your Strategy
                </h2>
                <p className="text-sm max-w-md" style={{ color: colors.textMuted }}>
                  Describe your investment goals, risk tolerance, and preferences.
                  I'll create a personalized portfolio strategy for you.
                </p>
                <div className="grid grid-cols-2 gap-3 mt-6">
                  {suggestedPrompts.map((prompt) => {
                    const Icon = prompt.icon;
                    return (
                      <button
                        key={prompt.title}
                        className="flex items-start gap-3 p-3 rounded-xl text-left transition-colors"
                        style={{ backgroundColor: colors.bgMint, border: `1px solid ${colors.border}` }}
                        onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = colors.bgHover)}
                        onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = colors.bgMint)}
                        onClick={() => setInputValue(prompt.title)}
                      >
                        <Icon className="w-5 h-5 flex-shrink-0 mt-0.5" style={{ color: colors.accent }} />
                        <div>
                          <p className="text-sm font-medium" style={{ color: colors.textPrimary }}>
                            {prompt.title}
                          </p>
                          <p className="text-xs mt-0.5" style={{ color: colors.textMuted }}>
                            {prompt.description}
                          </p>
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>
            ) : (
              <div className="space-y-6">
                {messages.map((message, index) => (
                  <div key={index}>
                    {message.role === "user" ? (
                      <div className="flex justify-end">
                        <div
                          className="max-w-[80%] px-4 py-3 rounded-2xl rounded-br-md"
                          style={{ backgroundColor: colors.accent, color: "#FFFFFF" }}
                        >
                          <p className="text-sm">{message.content}</p>
                        </div>
                      </div>
                    ) : (
                      <div>
                        <div className="flex items-start gap-3">
                          <div
                            className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0"
                            style={{ background: `linear-gradient(135deg, ${colors.accent}, ${colors.gain})` }}
                          >
                            <Sparkles className="w-4 h-4 text-white" />
                          </div>
                          <div className="flex-1">
                            <div
                              className="px-4 py-3 rounded-2xl rounded-tl-md"
                              style={{ backgroundColor: colors.bgMint }}
                            >
                              <p className="text-sm whitespace-pre-line" style={{ color: colors.textPrimary }}>
                                {message.content}
                              </p>

                              {message.stocks && (
                                <div className="mt-4 space-y-2">
                                  {message.stocks.map((stock) => (
                                    <div
                                      key={stock.symbol}
                                      className="flex items-center justify-between p-3 rounded-lg"
                                      style={{ backgroundColor: colors.bg, border: `1px solid ${colors.border}` }}
                                    >
                                      <div className="flex items-center gap-3">
                                        <div
                                          className="w-10 h-10 rounded-lg flex items-center justify-center"
                                          style={{ backgroundColor: colors.bgMint }}
                                        >
                                          <span className="text-xs font-bold" style={{ color: colors.textPrimary }}>
                                            {stock.symbol.slice(0, 2)}
                                          </span>
                                        </div>
                                        <div>
                                          <p className="text-sm font-medium" style={{ color: colors.textPrimary }}>
                                            {stock.symbol}
                                          </p>
                                          <p className="text-xs" style={{ color: colors.textMuted }}>
                                            {stock.reason}
                                          </p>
                                        </div>
                                      </div>
                                      <div
                                        className="px-3 py-1 rounded-full text-sm font-semibold"
                                        style={{ backgroundColor: colors.gainBg, color: colors.gain }}
                                      >
                                        {stock.allocation}%
                                      </div>
                                    </div>
                                  ))}
                                </div>
                              )}
                            </div>

                            <div className="flex items-center gap-2 mt-2">
                              <button className="p-1.5 rounded-lg" style={{ color: colors.textMuted }}>
                                <Copy className="w-4 h-4" />
                              </button>
                              <button className="p-1.5 rounded-lg" style={{ color: colors.textMuted }}>
                                <ThumbsUp className="w-4 h-4" />
                              </button>
                              <button className="p-1.5 rounded-lg" style={{ color: colors.textMuted }}>
                                <ThumbsDown className="w-4 h-4" />
                              </button>
                              <button className="p-1.5 rounded-lg" style={{ color: colors.textMuted }}>
                                <RotateCcw className="w-4 h-4" />
                              </button>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                ))}

                {isTyping && (
                  <div className="flex items-center gap-3">
                    <div
                      className="w-8 h-8 rounded-lg flex items-center justify-center"
                      style={{ background: `linear-gradient(135deg, ${colors.accent}, ${colors.gain})` }}
                    >
                      <Sparkles className="w-4 h-4 text-white" />
                    </div>
                    <div className="flex items-center gap-1">
                      <span className="w-2 h-2 rounded-full animate-bounce" style={{ backgroundColor: colors.textMuted, animationDelay: "0ms" }} />
                      <span className="w-2 h-2 rounded-full animate-bounce" style={{ backgroundColor: colors.textMuted, animationDelay: "150ms" }} />
                      <span className="w-2 h-2 rounded-full animate-bounce" style={{ backgroundColor: colors.textMuted, animationDelay: "300ms" }} />
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Input Area */}
          <div
            className="mt-4 rounded-xl p-3 flex items-center gap-3"
            style={{ backgroundColor: colors.bg, border: `1px solid ${colors.border}` }}
          >
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSend()}
              placeholder="Describe your investment strategy or ask a question..."
              className="flex-1 text-sm focus:outline-none"
              style={{ color: colors.textPrimary }}
            />
            <button
              onClick={handleSend}
              disabled={!inputValue.trim()}
              className="p-2.5 rounded-xl transition-opacity"
              style={{
                background: `linear-gradient(135deg, ${colors.accent}, ${colors.gain})`,
                opacity: inputValue.trim() ? 1 : 0.5,
              }}
            >
              <Send className="w-4 h-4 text-white" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
