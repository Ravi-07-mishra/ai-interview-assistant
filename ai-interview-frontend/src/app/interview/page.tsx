// app/interview/page.tsx
"use client";

import { useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";

export default function InterviewPage() {
  const API = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:4000";
  const search = useSearchParams();
  const dataq = search?.get("data") ?? "";
  const [parsed, setParsed] = useState<any>(null);
  const [question, setQuestion] = useState<string>("Explain ACID properties.");
  const [answer, setAnswer] = useState("");
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!dataq) return;
    try {
      setParsed(JSON.parse(decodeURIComponent(dataq)));
      // optional: create a question based on skills
      const p = JSON.parse(decodeURIComponent(dataq));
      if (p?.skills && p.skills.length > 0) {
        setQuestion(`Describe a project where you used ${p.skills[0]}. What challenges did you face and how did you solve them?`);
      }
    } catch (e) {
      setParsed(null);
    }
  }, [dataq]);

  async function submitAnswer() {
    setLoading(true);
    setResult(null);

    try {
      const payload = {
        question_text: question,
        candidate_answer: answer,
        ideal_outline: ""
      };

      const res = await fetch(`${API}/evaluate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const j = await res.json();
      if (!res.ok) throw new Error(j.error || JSON.stringify(j));
      setResult(j);
    } catch (err: any) {
      setResult({ error: err.message || String(err) });
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-gray-900 text-white p-6">
      <div className="max-w-3xl mx-auto bg-gray-800 p-6 rounded-lg shadow">
        <h1 className="text-2xl font-bold mb-4">Interview — Demo</h1>

        {parsed ? (
          <div className="mb-4">
            <div className="text-sm text-gray-300">Candidate</div>
            <div className="font-medium">{parsed.name ?? "Unknown"}</div>
            <div className="text-sm text-gray-400">{parsed.email ?? ""} {parsed.phone ? `· ${parsed.phone}` : ""}</div>
            <div className="text-sm text-gray-300 mt-2">Detected skills</div>
            <div className="flex gap-2 mt-1">
              {Array.isArray(parsed.skills) && parsed.skills.length ? parsed.skills.map((s:string) => (
                <span key={s} className="px-2 py-1 bg-gray-700 rounded text-sm">{s}</span>
              )) : <span className="text-gray-400">—</span>}
            </div>
          </div>
        ) : (
          <div className="mb-4 text-gray-300">No resume passed — this is a demo. You can upload one at <a className="text-blue-400" href="/resume">/resume</a></div>
        )}

        <div className="mb-4">
          <h2 className="font-semibold">Question</h2>
          <p className="text-gray-300 mt-1">{question}</p>
        </div>

        <textarea
          value={answer}
          onChange={(e) => setAnswer(e.target.value)}
          rows={6}
          className="w-full p-3 rounded bg-gray-700 border border-gray-600 focus:outline-none"
          placeholder="Type your answer here..."
        />

        <div className="flex gap-3 mt-4">
          <button
            onClick={submitAnswer}
            disabled={!answer || loading}
            className={`px-4 py-2 rounded ${!answer || loading ? "bg-gray-600" : "bg-blue-600 hover:bg-blue-700"}`}
          >
            {loading ? "Evaluating..." : "Submit Answer"}
          </button>

          <button
            onClick={() => { setAnswer(""); setResult(null); }}
            className="px-4 py-2 rounded bg-gray-700 hover:bg-gray-600"
          >
            Clear
          </button>
        </div>

        {result && (
          <div className="mt-6 bg-gray-700 p-4 rounded">
            <h3 className="font-semibold">Evaluation</h3>
            <pre className="text-sm whitespace-pre-wrap mt-2">{JSON.stringify(result, null, 2)}</pre>
          </div>
        )}
      </div>
    </main>
  );
}
