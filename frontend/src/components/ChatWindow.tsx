"use client";

import { useState, useRef, useEffect } from "react";
import { Send, Trash2, Loader2 } from "lucide-react";
import { queryRAGStream, clearHistory } from "@/lib/api";

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: any[];
}

export function ChatWindow() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const question = input.trim();
    setInput("");

    // Add user message
    setMessages((prev) => [...prev, { role: "user", content: question }]);
    setIsLoading(true);

    // Add empty assistant message (will be filled by streaming)
    setMessages((prev) => [...prev, { role: "assistant", content: "" }]);

    await queryRAGStream(
      question,
      // onToken — append each token to the last message
      (token) => {
        setMessages((prev) => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last.role === "assistant") {
            last.content += token;
          }
          return [...updated];
        });
      },
      // onSources — attach sources to the last message
      (sources) => {
        setMessages((prev) => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last.role === "assistant") {
            last.sources = sources;
          }
          return [...updated];
        });
      },
      // onDone
      () => {
        setIsLoading(false);
      },
      // onError
      (error) => {
        setMessages((prev) => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last.role === "assistant") {
            last.content = `Error: ${error}`;
          }
          return [...updated];
        });
        setIsLoading(false);
      }
    );
  };

  const handleClearHistory = async () => {
    try {
      await clearHistory();
      setMessages([]);
    } catch {
      // Ignore errors on clear
    }
  };

  return (
    <div className="flex flex-col h-[calc(100vh-4rem)]">
      {/* Messages area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="flex items-center justify-center h-full text-gray-400">
            <div className="text-center">
              <p className="text-lg">Ask a question about your documents</p>
              <p className="text-sm mt-2">
                Upload PDFs first, then ask anything about them.
              </p>
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[80%] rounded-2xl px-4 py-3 ${
                msg.role === "user"
                  ? "bg-blue-600 text-white"
                  : "bg-white border shadow-sm"
              }`}
            >
              <p className="whitespace-pre-wrap">{msg.content}</p>

              {/* Source citations */}
              {msg.sources && msg.sources.length > 0 && (
                <div className="mt-3 pt-3 border-t border-gray-200">
                  <p className="text-xs font-semibold text-gray-500 mb-1">
                    Sources:
                  </p>
                  {msg.sources.map((src, j) => (
                    <div
                      key={j}
                      className="text-xs text-gray-500 flex items-center gap-1 mt-1"
                    >
                      <span className="font-medium">
                        [{src.filename}, p.{src.page}]
                      </span>
                      <span className="text-gray-400">
                        score: {src.final_score.toFixed(3)}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}

        {isLoading && messages[messages.length - 1]?.content === "" && (
          <div className="flex justify-start">
            <div className="bg-white border rounded-2xl px-4 py-3 shadow-sm">
              <Loader2 className="w-5 h-5 animate-spin text-gray-400" />
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input area */}
      <div className="border-t bg-white p-4">
        <form onSubmit={handleSubmit} className="flex gap-2 max-w-4xl mx-auto">
          <button
            type="button"
            onClick={handleClearHistory}
            className="p-3 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded-lg transition"
            title="Clear conversation"
          >
            <Trash2 className="w-5 h-5" />
          </button>

          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question about your documents..."
            className="flex-1 px-4 py-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            disabled={isLoading}
          />

          <button
            type="submit"
            disabled={!input.trim() || isLoading}
            className="p-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition"
          >
            <Send className="w-5 h-5" />
          </button>
        </form>
      </div>
    </div>
  );
}
