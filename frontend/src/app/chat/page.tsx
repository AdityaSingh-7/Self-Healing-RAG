"use client";

import { useState, useRef, useEffect } from "react";
import { Send, Loader2, Shield, AlertTriangle, CheckCircle, RefreshCw } from "lucide-react";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

interface HealingReport {
  attempts: Array<{
    attempt: number;
    strategy: string;
    confidence: number;
    reason: string;
  }>;
  original_confidence: number;
  final_confidence: number;
  healed: boolean;
  strategy_used: string | null;
}

interface Message {
  role: "user" | "assistant";
  content: string;
  confidence?: number;
  healed?: boolean;
  degraded?: boolean;
  attempts?: number;
  strategy_used?: string | null;
  sources?: any[];
  healing_report?: HealingReport;
  latency_ms?: number;
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [showReport, setShowReport] = useState<number | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const question = input.trim();
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: question }]);
    setIsLoading(true);

    try {
      const response = await fetch(`${BACKEND_URL}/healing/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, top_k: 5, recency_weight: 0.2, max_attempts: 3 }),
      });

      const data = await response.json();

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.answer,
          confidence: data.confidence,
          healed: data.healed,
          degraded: data.degraded,
          attempts: data.attempts,
          strategy_used: data.strategy_used,
          sources: data.sources,
          healing_report: data.healing_report,
          latency_ms: data.latency_ms,
        },
      ]);
    } catch (err: any) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Error: ${err.message || "Request failed"}` },
      ]);
    }

    setIsLoading(false);
  };

  const getConfidenceBadge = (msg: Message) => {
    if (!msg.confidence) return null;
    const conf = msg.confidence;

    if (conf >= 0.8) {
      return (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-green-100 text-green-700 rounded-full text-xs font-medium">
          <CheckCircle className="w-3 h-3" />
          {(conf * 100).toFixed(0)}% confident
        </span>
      );
    } else if (conf >= 0.5) {
      return (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-yellow-100 text-yellow-700 rounded-full text-xs font-medium">
          <AlertTriangle className="w-3 h-3" />
          {(conf * 100).toFixed(0)}% confident
        </span>
      );
    } else {
      return (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-red-100 text-red-700 rounded-full text-xs font-medium">
          <AlertTriangle className="w-3 h-3" />
          {(conf * 100).toFixed(0)}% confident
        </span>
      );
    }
  };

  return (
    <div className="flex flex-col h-[calc(100vh-4rem)]">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="flex items-center justify-center h-full text-gray-400">
            <div className="text-center">
              <Shield className="w-12 h-12 mx-auto mb-3 text-gray-300" />
              <p className="text-lg">Ask a question with self-healing</p>
              <p className="text-sm mt-2">
                The system validates every answer and retries if confidence is low.
              </p>
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
            <div className={`max-w-[85%] rounded-2xl px-4 py-3 ${
              msg.role === "user"
                ? "bg-blue-600 text-white"
                : msg.degraded
                ? "bg-yellow-50 border border-yellow-200"
                : "bg-white border shadow-sm"
            }`}>
              {/* Answer text */}
              <p className="whitespace-pre-wrap">{msg.content}</p>

              {/* Healing metadata */}
              {msg.role === "assistant" && msg.confidence !== undefined && (
                <div className="mt-3 pt-3 border-t border-gray-100 space-y-2">
                  {/* Confidence + healing badges */}
                  <div className="flex items-center gap-2 flex-wrap">
                    {getConfidenceBadge(msg)}

                    {msg.healed && (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-blue-100 text-blue-700 rounded-full text-xs font-medium">
                        <RefreshCw className="w-3 h-3" />
                        Healed: {msg.strategy_used}
                      </span>
                    )}

                    {msg.attempts && msg.attempts > 1 && (
                      <span className="text-xs text-gray-400">
                        {msg.attempts} attempts
                      </span>
                    )}

                    {msg.latency_ms && (
                      <span className="text-xs text-gray-400">
                        {(msg.latency_ms / 1000).toFixed(1)}s
                      </span>
                    )}
                  </div>

                  {/* Sources */}
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="space-y-1">
                      <p className="text-xs font-semibold text-gray-500">Sources:</p>
                      {msg.sources.slice(0, 3).map((src: any, j: number) => (
                        <p key={j} className="text-xs text-gray-500">
                          [{src.filename}, p.{src.page}] — {(src.final_score * 100).toFixed(0)}% match
                        </p>
                      ))}
                    </div>
                  )}

                  {/* Healing report toggle */}
                  {msg.healing_report && msg.healing_report.attempts.length > 1 && (
                    <button
                      onClick={() => setShowReport(showReport === i ? null : i)}
                      className="text-xs text-blue-500 hover:underline"
                    >
                      {showReport === i ? "Hide" : "Show"} healing report
                    </button>
                  )}

                  {/* Expanded healing report */}
                  {showReport === i && msg.healing_report && (
                    <div className="mt-2 p-3 bg-gray-50 rounded-lg text-xs space-y-2">
                      <p className="font-semibold text-gray-700">Healing Report:</p>
                      {msg.healing_report.attempts.map((attempt, j) => (
                        <div key={j} className="flex items-start gap-2">
                          <span className={`w-2 h-2 rounded-full mt-1 flex-shrink-0 ${
                            attempt.confidence >= 0.8 ? "bg-green-500" :
                            attempt.confidence >= 0.5 ? "bg-yellow-500" : "bg-red-500"
                          }`} />
                          <div>
                            <span className="text-gray-700">
                              Attempt {attempt.attempt}: <strong>{attempt.strategy}</strong> → {(attempt.confidence * 100).toFixed(0)}%
                            </span>
                            <p className="text-gray-400">{attempt.reason}</p>
                          </div>
                        </div>
                      ))}
                      <div className="pt-2 border-t border-gray-200 text-gray-500">
                        Confidence: {((msg.healing_report.original_confidence || 0) * 100).toFixed(0)}% → {((msg.healing_report.final_confidence || msg.confidence || 0) * 100).toFixed(0)}%
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-white border rounded-2xl px-4 py-3 shadow-sm flex items-center gap-2">
              <Loader2 className="w-4 h-4 animate-spin text-blue-500" />
              <span className="text-sm text-gray-500">Validating & healing...</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="border-t bg-white p-4">
        <form onSubmit={handleSubmit} className="flex gap-2 max-w-4xl mx-auto">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question (self-healing enabled)..."
            className="flex-1 px-4 py-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={!input.trim() || isLoading}
            className="p-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 transition"
          >
            <Send className="w-5 h-5" />
          </button>
        </form>
      </div>
    </div>
  );
}
