"use client";

import { useAuth } from "../context/AuthContext";

const API = process.env.NEXT_PUBLIC_BACKEND_URL;

export type DashboardData = {
  user: {
    name: string;
    email: string;
    role: string;
  };
  stats: {
    totalInterviews: number;
    averageScore: number | null;
  };
  pastSessions: {
    sessionId: string;
    startedAt: string;
    endedAt?: string;
    status: string;
    qas: {
      question: string;
      score: number | null;
      feedback: string | null;
    }[];
  }[];
};

export function useProfile() {
  const { token } = useAuth();

  async function fetchDashboard(): Promise<DashboardData> {
    if (!token) {
      throw new Error("Missing auth token");
    }

    const res = await fetch(`${API}/profile/dashboard`, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });

    if (!res.ok) {
      throw new Error("Failed to fetch dashboard");
    }

    return res.json();
  }

  return { fetchDashboard };
}
