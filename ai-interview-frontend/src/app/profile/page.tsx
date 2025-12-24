"use client";

import { useEffect, useState } from "react";
import { useProfile, DashboardData } from "../hooks/useProfile";
import { useAuth } from "../context/AuthContext";

export default function ProfilePage() {
  const { fetchDashboard } = useProfile();
  const { token } = useAuth();

  const [data, setData] = useState<DashboardData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);


  useEffect(() => {
    if (!token) return;

    fetchDashboard()
      .then(setData)
      .catch((err) => {
        console.error(err);
        setError("Unable to load profile dashboard");
      })
      .finally(() => setLoading(false));
  }, [token]);

  if (!token) return <p className="p-6">Not logged in</p>;
  if (loading) return <p className="p-6">Loading dashboard…</p>;
  if (error) return <p className="p-6 text-red-600">{error}</p>;
  if (!data) return null;

  const { user, stats, pastSessions } = data;

  return (
    <div className="min-h-screen bg-gray-50">
      {/* ===== TOP BAR ===== */}
      <div className="bg-white border-b px-8 py-4 flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold">Profile Dashboard</h1>
          <p className="text-sm text-gray-500">
            Interview analytics & performance insights
          </p>
        </div>

        {/* User info (top right) */}
        <div className="text-right">
          <p className="font-semibold">{user.name}</p>
          <p className="text-sm text-gray-500">{user.email}</p>
          <span className="text-xs bg-indigo-100 text-indigo-700 px-2 py-1 rounded">
            {user.role}
          </span>
        </div>
      </div>

      {/* ===== STATS ===== */}
      <div className="p-8 grid grid-cols-1 md:grid-cols-3 gap-6">
        <StatCard
          title="Total Interviews"
          value={stats.totalInterviews}
        />
        <StatCard
          title="Average Score"
          value={stats.averageScore ?? "N/A"}
        />
        <StatCard
          title="Best Score"
          value={
            pastSessions.length
              ? Math.max(
                  ...pastSessions.flatMap((s: any) =>
                    s.qas.map((q: any) => q.score || 0)
                  )
                ).toFixed(2)
              : "N/A"
          }
        />
      </div>

      {/* ===== PAST SESSIONS ===== */}
      <div className="px-8 pb-10">
        <h2 className="text-xl font-semibold mb-4">
          Past Interview Sessions
        </h2>

        {pastSessions.length === 0 ? (
          <div className="bg-white p-6 rounded shadow text-gray-500">
            No interviews yet
          </div>
        ) : (
          <div className="space-y-4">
            {pastSessions.map((s: any, idx: number) => (
              <div
                key={idx}
                className="bg-white p-6 rounded shadow"
              >
                <div className="flex justify-between mb-3">
                  <p className="font-semibold">
                    Session #{idx + 1}
                  </p>
                  <p className="text-sm text-gray-500">
                    {new Date(s.startedAt).toLocaleString()}
                  </p>
                </div>

                <div className="space-y-2">
                  {s.qas.map((q: any, i: number) => (
                    <div
                      key={i}
                      className="border rounded p-3 text-sm"
                    >
                      <p className="font-medium">
                        Q{i + 1}: {q.question}
                      </p>
                      <p className="text-gray-600">
                        Score:{" "}
                        <span className="font-semibold">
                          {q.score ?? "N/A"}
                        </span>
                      </p>
                      {q.feedback && (
                        <p className="text-gray-500 mt-1">
                          Feedback: {q.feedback}
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

/* ===== Small reusable stat card ===== */
function StatCard({ title, value }: any) {
  return (
    <div className="bg-white rounded shadow p-6">
      <p className="text-sm text-gray-500">{title}</p>
      <p className="text-3xl font-bold mt-2">{value}</p>
    </div>
  );
}
