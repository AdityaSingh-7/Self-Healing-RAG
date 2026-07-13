"use client";

import { useState, useEffect } from "react";
import { Activity, DollarSign, Zap, Target, RefreshCw, TrendingUp } from "lucide-react";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

export default function DashboardPage() {
  const [strategies, setStrategies] = useState<any>(null);
  const [costs, setCosts] = useState<any>(null);
  const [history, setHistory] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    setLoading(true);
    try {
      const [stratRes, costRes, histRes] = await Promise.all([
        fetch(`${BACKEND_URL}/healing/strategies`).then((r) => r.json()),
        fetch(`${BACKEND_URL}/healing/costs`).then((r) => r.json()),
        fetch(`${BACKEND_URL}/healing/history?limit=10`).then((r) => r.json()),
      ]);
      setStrategies(stratRes);
      setCosts(costRes);
      setHistory(histRes.events || []);
    } catch (err) {
      console.error("Failed to fetch dashboard data:", err);
    }
    setLoading(false);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[calc(100vh-4rem)]">
        <RefreshCw className="w-8 h-8 animate-spin text-gray-400" />
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-8">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Healing Dashboard</h1>
        <button
          onClick={fetchData}
          className="flex items-center gap-2 px-3 py-2 text-sm border rounded-lg hover:bg-gray-50 transition"
        >
          <RefreshCw className="w-4 h-4" />
          Refresh
        </button>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatCard
          icon={<Target className="w-5 h-5 text-green-600" />}
          label="Healing Success Rate"
          value={strategies?.overall?.healing_success_rate
            ? `${(strategies.overall.healing_success_rate * 100).toFixed(0)}%`
            : "—"}
        />
        <StatCard
          icon={<Activity className="w-5 h-5 text-blue-600" />}
          label="Total Healing Attempts"
          value={strategies?.overall?.total_healing_attempts || 0}
        />
        <StatCard
          icon={<TrendingUp className="w-5 h-5 text-purple-600" />}
          label="Avg Confidence Gain"
          value={strategies?.overall?.avg_confidence_improvement
            ? `+${(strategies.overall.avg_confidence_improvement * 100).toFixed(1)}%`
            : "—"}
        />
        <StatCard
          icon={<DollarSign className="w-5 h-5 text-orange-600" />}
          label="Avg Cost/Query"
          value={costs?.summary?.avg_cost_per_query_usd
            ? `$${costs.summary.avg_cost_per_query_usd.toFixed(5)}`
            : "—"}
        />
      </div>

      {/* Strategy Performance */}
      <div className="bg-white rounded-xl border p-6">
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Zap className="w-5 h-5 text-yellow-500" />
          Strategy Performance
        </h2>

        {strategies?.strategies?.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b text-left text-gray-500">
                  <th className="pb-3 pr-4">Strategy</th>
                  <th className="pb-3 pr-4">Attempts</th>
                  <th className="pb-3 pr-4">Successes</th>
                  <th className="pb-3 pr-4">Success Rate</th>
                  <th className="pb-3">Avg Improvement</th>
                </tr>
              </thead>
              <tbody>
                {strategies.strategies.map((s: any, i: number) => (
                  <tr key={i} className="border-b last:border-0">
                    <td className="py-3 pr-4 font-medium">{s.strategy_name}</td>
                    <td className="py-3 pr-4">{s.total_attempts}</td>
                    <td className="py-3 pr-4">{s.total_successes}</td>
                    <td className="py-3 pr-4">
                      <div className="flex items-center gap-2">
                        <div className="w-24 h-2 bg-gray-200 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-green-500 rounded-full"
                            style={{ width: `${(s.success_rate || 0) * 100}%` }}
                          />
                        </div>
                        <span>{((s.success_rate || 0) * 100).toFixed(0)}%</span>
                      </div>
                    </td>
                    <td className="py-3">
                      <span className={s.avg_improvement > 0 ? "text-green-600" : "text-gray-400"}>
                        {s.avg_improvement > 0 ? "+" : ""}{((s.avg_improvement || 0) * 100).toFixed(1)}%
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-gray-400 text-center py-8">
            No healing data yet. Ask questions in the chat to generate stats.
          </p>
        )}
      </div>

      {/* Cost Comparison */}
      {costs?.comparison && (
        <div className="bg-white rounded-xl border p-6">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <DollarSign className="w-5 h-5 text-green-600" />
            Cost Comparison: Standard vs Healing
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="p-4 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-500 mb-1">Standard RAG</p>
              <p className="text-2xl font-bold">{costs.comparison.standard_queries?.avg_tokens || 0}</p>
              <p className="text-xs text-gray-400">avg tokens/query</p>
            </div>
            <div className="p-4 bg-blue-50 rounded-lg">
              <p className="text-sm text-blue-600 mb-1">Self-Healing RAG</p>
              <p className="text-2xl font-bold">{costs.comparison.healed_queries?.avg_tokens || 0}</p>
              <p className="text-xs text-gray-400">avg tokens/query</p>
            </div>
            <div className="p-4 bg-purple-50 rounded-lg">
              <p className="text-sm text-purple-600 mb-1">Cost Multiplier</p>
              <p className="text-2xl font-bold">{costs.comparison.comparison?.token_multiplier || "—"}</p>
              <p className="text-xs text-gray-400">tokens (healing vs standard)</p>
            </div>
          </div>
        </div>
      )}

      {/* Recent Healing Events */}
      <div className="bg-white rounded-xl border p-6">
        <h2 className="text-lg font-semibold mb-4">Recent Healing Events</h2>

        {history.length > 0 ? (
          <div className="space-y-3">
            {history.map((event, i) => (
              <div key={i} className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg text-sm">
                <span className={`w-2 h-2 rounded-full flex-shrink-0 ${
                  event.success ? "bg-green-500" : "bg-red-500"
                }`} />
                <div className="flex-1 min-w-0">
                  <p className="text-gray-700 truncate">{event.question}</p>
                  <p className="text-xs text-gray-400">
                    {event.strategy_name} · {(event.confidence_before * 100).toFixed(0)}% → {(event.confidence_after * 100).toFixed(0)}%
                  </p>
                </div>
                <span className={`text-xs font-medium ${
                  event.success ? "text-green-600" : "text-red-600"
                }`}>
                  {event.success ? "Healed" : "Failed"}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-gray-400 text-center py-8">
            No healing events yet.
          </p>
        )}
      </div>
    </div>
  );
}

function StatCard({ icon, label, value }: { icon: React.ReactNode; label: string; value: any }) {
  return (
    <div className="bg-white rounded-xl border p-4">
      <div className="flex items-center gap-3">
        {icon}
        <div>
          <p className="text-xs text-gray-500">{label}</p>
          <p className="text-xl font-bold">{value}</p>
        </div>
      </div>
    </div>
  );
}
