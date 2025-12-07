"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "../context/AuthContext";
import ResumeUploader from "./ResumeUploader";

export default function ResumePage() {
  const router = useRouter();
  const { token, loading } = useAuth();

  useEffect(() => {
    if (loading) return; // Wait until localStorage loads

    if (!token) {
      router.replace("/auth/login");
    }
  }, [token, loading, router]);

  if (loading) {
    return <div className="p-10">Checking authentication...</div>;
  }

  if (!token) {
    return <div className="p-10">Redirecting...</div>;
  }

  return <ResumeUploader />;
}
