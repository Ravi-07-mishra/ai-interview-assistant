"use client";

import { useState } from "react";

export default function Home() {
  const API = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:4000";

  const [question] = useState("Explain ACID properties.");
  const [answer, setAnswer] = useState("");
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  async function submitAnswer() {
    setLoading(true);
    setResult(null);

    try {
      const res = await fetch(`${API}/evaluate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question_text: question,
          candidate_answer: answer,
          ideal_outline: "atomicity, consistency, isolation, durability"
        })
      });

      const json = await res.json();
      setResult(json);
    } catch (error) {
      setResult({ error: String(error) });
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-gray-900 text-white p-6 flex flex-col items-center">
      <div className="w-full max-w-2xl bg-gray-800 p-6 rounded-xl shadow-lg">
        <h1 className="text-3xl font-bold mb-4 text-center">
          AI Interview Assistant
        </h1>

        {/* Question */}
        <div className="mb-4">
          <h2 className="font-semibold text-lg">Question:</h2>
          <p className="text-gray-300 mt-1">{question}</p>
        </div>

        {/* Answer Input */}
        <textarea
          value={answer}
          onChange={(e) => setAnswer(e.target.value)}
          placeholder="Type your answer here..."
          className="w-full p-3 rounded-lg bg-gray-700 border border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
          rows={6}
        />

        {/* Submit Button */}
        <button
          onClick={submitAnswer}
          disabled={loading || !answer}
          className={`w-full mt-4 py-3 rounded-lg font-semibold transition 
          ${loading || !answer
              ? "bg-gray-600 cursor-not-allowed"
              : "bg-blue-600 hover:bg-blue-700"
            }`}
        >
          {loading ? "Evaluating..." : "Submit Answer"}
        </button>

        {/* Result */}
        {result && (
          <div className="mt-6 bg-gray-700 p-4 rounded-lg">
            <h3 className="font-semibold text-xl mb-2">Evaluation Result</h3>
            <pre className="text-gray-200 whitespace-pre-wrap text-sm">
              {JSON.stringify(result, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </main>
  );
}
