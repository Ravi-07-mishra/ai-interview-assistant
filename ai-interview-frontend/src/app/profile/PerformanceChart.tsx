"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";

type Props = {
  sessions: any[];
};

export default function PerformanceChart({ sessions }: Props) {
  const chartData = sessions.map((session, index) => ({
    session: `S${index + 1}`,

    Screening: session.rounds?.screening?.averageScore ?? null,
    Technical: session.rounds?.technical?.averageScore ?? null,
    Behavioral: session.rounds?.behavioral?.averageScore ?? null,
  }));

  return (
    <div className="bg-gradient-to-br from-indigo-50 to-white rounded-xl shadow-sm p-6 border border-indigo-100">
      <h3 className="text-lg font-semibold text-indigo-900 mb-4">
        Performance Trend Across Sessions
      </h3>

      <div className="w-full h-[300px]">
        <ResponsiveContainer width="100%" height="100%">
  <LineChart
    data={chartData}
    margin={{ top: 20, right: 30, left: 10, bottom: 30 }}
  >
    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />

    {/* X Axis */}
    <XAxis
  dataKey="session"
  interval={Math.ceil(chartData.length / 6)}
  tick={{ fontSize: 12 }}
/>


    {/* Y Axis */}
    <YAxis
  domain={[0.5, 1]}
  ticks={[0.5, 0.6, 0.7, 0.8, 0.9, 1]}
  tickFormatter={(v) => v.toFixed(2)}
/>


    <Tooltip />
    <Legend />

    <Line
  type="monotone"
  dataKey="Screening"
  stroke="#2563eb"
  strokeWidth={3}
  dot={{ r: 3 }}
  connectNulls
/>

<Line
  type="monotone"
  dataKey="Technical"
  stroke="#111827"
  strokeWidth={3}
  dot={{ r: 3 }}
  connectNulls
/>

<Line
  type="monotone"
  dataKey="Behavioral"
  stroke="#dc2626"
  strokeWidth={3}
  dot={{ r: 3 }}
  connectNulls
/>

  </LineChart>
</ResponsiveContainer>

      </div>
    </div>
  );
}
