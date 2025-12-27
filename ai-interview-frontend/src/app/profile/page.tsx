"use client";

import { useEffect, useState } from "react";
import { useProfile, DashboardData } from "../hooks/useProfile";
import { useAuth } from "../context/AuthContext";
import PerformanceChart from "./PerformanceChart";

/* ================= TYPES ================= */

type RoundSummary = {
  averageScore: number | null;
  feedback: string | null;
};

type SessionSummary = {
  sessionId: string;
  date: string;
  rounds: {
    screening: RoundSummary | null;
    technical: RoundSummary | null;
    behavioral: RoundSummary | null;
  };
};

export default function ProfilePage() {
  const { fetchDashboard } = useProfile();
  const { token } = useAuth();

  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!token) return;

    fetchDashboard()
      .then(setData)
      .catch(() => setError("Unable to load profile dashboard"))
      .finally(() => setLoading(false));
  }, [token]);

  if (!token) return <p className="p-8">Not logged in</p>;
  if (loading) return <p className="p-8">Loading dashboard…</p>;
  if (error) return <p className="p-8 text-red-600">{error}</p>;
  if (!data) return null;

  const { user, stats, interviewHistory } = data;

  const bestScore =
    interviewHistory.length > 0
      ? Math.max(
          ...interviewHistory.flatMap((s: SessionSummary) =>
            Object.values(s.rounds)
              .filter(Boolean)
              .map(r => r!.averageScore ?? 0)
          )
        ).toFixed(2)
      : "N/A";

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-sky-50 to-purple-50">

      {/* ================= TOP BAR ================= */}
      <div className="sticky top-0 z-20 bg-white/80 backdrop-blur border-b border-indigo-100 px-10 py-5 flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 tracking-tight">
            Profile Dashboard
          </h1>
          <p className="text-sm text-gray-500 mt-1">
            Interview analytics & performance insights
          </p>
        </div>

        <div className="flex items-center gap-4">
          <div className="text-right">
            <p className="font-semibold text-gray-900">{user.name}</p>
            <p className="text-sm text-gray-500">{user.email}</p>
          </div>

          <div className="h-10 w-10 rounded-full bg-indigo-600 text-white flex items-center justify-center font-bold uppercase">
            {user.name?.[0]}
          </div>
        </div>
      </div>

      {/* ================= STATS ================= */}
      <div className="px-10 pt-10 grid grid-cols-1 md:grid-cols-3 gap-6">
        <StatCard title="Total Interviews" value={stats.totalInterviews} />
        <StatCard title="Average Score" value={stats.averageScore} />
        <StatCard title="Best Score" value={bestScore} />
      </div>

      {/* ================= PERFORMANCE CHART ================= */}
      <div className="px-10 mt-10">
        <div className="bg-white rounded-2xl border border-indigo-100 shadow-sm p-6">
          <h2 className="text-lg font-semibold text-indigo-900 mb-4">
            Performance Trend Across Sessions
          </h2>

          <PerformanceChart sessions={interviewHistory} />
        </div>
      </div>

      {/* ================= PAST SESSIONS ================= */}
      <div className="px-10 pb-16 mt-12">
        <h2 className="text-2xl font-semibold mb-6 text-gray-900">
          Past Interview Sessions
        </h2>

        {interviewHistory.length === 0 ? (
          <div className="bg-white p-6 rounded-xl shadow text-gray-500">
            No interviews yet
          </div>
        ) : (
          <div className="space-y-8">
            {interviewHistory.map((session: SessionSummary, idx: number) => (
              <div
                key={session.sessionId}
                className="bg-white rounded-2xl border border-indigo-100 shadow-sm p-6"
              >
                <div className="flex justify-between items-center mb-6">
                  <p className="text-lg font-semibold text-gray-900">
                    Session #{idx + 1}
                  </p>
                  <p className="text-sm text-gray-500">
                    {new Date(session.date).toLocaleString()}
                  </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  {(["screening", "technical", "behavioral"] as const).map(
                    round => {
                      const r = session.rounds[round];
                      if (!r) return null;

                      return (
                        <div
                          key={round}
                          className="rounded-xl p-5 bg-gradient-to-br from-indigo-50 to-white border border-indigo-100"
                        >
                          <p className="font-semibold text-indigo-800 capitalize mb-2">
                            {round} round
                          </p>

                          <p className="text-sm mb-3">
                            <span className="text-gray-500">Score:</span>{" "}
                            <span className="font-bold text-indigo-700 text-lg">
                              {r.averageScore}
                            </span>
                          </p>

                          {r.feedback && (
                            <p className="text-sm text-gray-700 leading-relaxed">
                              <span className="font-medium">Feedback:</span>{" "}
                              {r.feedback}
                            </p>
                          )}
                        </div>
                      );
                    }
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

/* ================= STAT CARD ================= */
function StatCard({ title, value }: { title: string; value: any }) {
  return (
    <div className="relative bg-white rounded-2xl p-6 border border-indigo-100 shadow-sm hover:shadow-lg transition">
      <div className="absolute inset-x-0 top-0 h-1 bg-gradient-to-r from-indigo-500 to-purple-500 rounded-t-2xl" />
      <p className="text-sm text-gray-500">{title}</p>
      <p className="text-4xl font-extrabold mt-2 text-indigo-700">
        {value ?? "N/A"}
      </p>
    </div>
  );
}
