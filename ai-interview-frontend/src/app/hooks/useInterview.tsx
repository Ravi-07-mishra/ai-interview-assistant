// hooks/useInterview.tsx
"use client";

import { useCallback, useState } from "react";
import { useAuth } from "../context/AuthContext";

export type InterviewQuestion = {
  questionId?: string;
  questionText: string;
  target_project?: string;
  technology_focus?: string;
  expectedAnswerType?: "short" | "medium" | "code" | "architectural";
  difficulty?: "easy" | "medium" | "hard" | "expert";
  ideal_outline?: string;
  ideal_answer_outline?: string;
  red_flags?: string[];
  action_type?: string;
  confidence?: number;
    is_probe?: boolean;
  round?: string;
  // keep index signature so we can attach raw / coding_challenge etc.
  [k: string]: any;
};

export type InterviewAnswerResult = {
  score?: number;
  overall_score?: number;
  rubricScores?: Record<string, number>;
  dimension_scores?: Record<string, number>;
  confidence?: number;
  verdict?: "fail" | "weak" | "acceptable" | "strong" | "exceptional";
  rationale?: string;
  red_flags_detected?: string[];
  missing_elements?: string[];
  improvement?: string;
  follow_up_probe?: string | null;
  nextQuestion?: InterviewQuestion | null;
  ended?: boolean;
  decision?: any | null;
  in_gray_zone?: boolean;
  needs_human_review?: boolean;
  round_info?: {
    current: string;
    progress: Record<string, any>;
    section_counts?: Record<string, number>;
  };
  eliminated?: boolean;
  elimination_reason?: string | null;
  is_probe?: boolean;
};

export type PerformanceMetrics = {
  question_count: number;
  average_score: number;
  last_score: number | null;
  consecutive_fails: number;
  consecutive_wins: number;
  trend: "improving" | "declining" | "stable" | "insufficient_data" | "unknown";
  confidence: number;
  score_variance?: number;
};
const normalizeType = (t: any) => t || "conceptual";

const API = process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ?? "";

export function useInterview() {
  const { token } = useAuth();
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [stage, setStage] = useState<"idle" | "uploading" | "running" | "done">("idle");
  const [currentQuestion, setCurrentQuestion] = useState<InterviewQuestion | null>(null);

  // Store feedback and performance metrics
  const [lastFeedback, setLastFeedback] = useState<string | null>(null);
  const [performanceMetrics, setPerformanceMetrics] = useState<PerformanceMetrics | null>(null);

  const [history, setHistory] = useState<Array<{
    q: InterviewQuestion;
    a?: any;
    result?: InterviewAnswerResult;
  }>>([]);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [resumeParsed, setResumeParsed] = useState<any | null>(null);
  const [resumeFileUrl, setResumeFileUrl] = useState<string | null>(null);
  const [finalDecision, setFinalDecision] = useState<any | null>(null);

 // REPLACE THIS FUNCTION in hooks/useInterview.tsx

const onResumeReady = useCallback((parsed: any, fileUrl?: string | null) => {
    console.log("📄 Resume Loaded in Hook:", parsed);

    // 🚨 CRITICAL FIX: If the backend provided the rich context, 
    // overwrite the short summary with it immediately.
    // This ensures that startInterview and submitAnswer always use the full text.
    if (parsed && (parsed as any).full_context_for_prompt) {
        console.log("✅ Upgrading Resume Summary to Rich Context");
        parsed.summary = (parsed as any).full_context_for_prompt;
    }

    setResumeParsed(parsed);
    if (fileUrl) setResumeFileUrl(fileUrl);
    setStage("idle");
}, []);

  const buildConversationFromHistory = useCallback(() => {
    const conv: Array<{ role: "assistant" | "user"; text: string }> = [];
    for (const entry of history) {
      if (entry.q?.questionText) conv.push({ role: "assistant", text: entry.q.questionText });
      if (entry.a) conv.push({ role: "user", text: String(entry.a) });
    }
    return conv;
  }, [history]);

const buildQuestionHistory = useCallback(() => {
  return history.map(h => ({
    question: h.q.questionText,
    type: normalizeType(h.q.type),   // 🔧 FIX
    target_project: h.q.target_project || "general",
    score: h.result?.overall_score ?? h.result?.score ?? null
  }));
}, [history]);



  const getAuthHeaders = useCallback(() => {
    if (!token) return {};
    return { Authorization: `Bearer ${token}` };
  }, [token]);

  /**
   * reportViolation(reason)
   * - always POSTs to /interview/violation with { sessionId, reason }
   * - returns the parsed server response (or throws for network errors)
   * - caller should handle or ignore failures as appropriate
   */
  const reportViolation = useCallback(async (reason: string) => {
    if (!sessionId) {
      // No active session: return a consistent shape so callers can proceed gracefully.
      return { ok: false, error: "no_session", message: "No active sessionId" };
    }

    try {
      const res = await fetch(`${API}/interview/violation`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...getAuthHeaders() },
        body: JSON.stringify({
          sessionId,
          reason
        })
      });

      // Try to parse JSON body safely
      const body = await res.json().catch(() => ({}));

      // Return server body so UI can inspect / show messages if needed.
      return body;
    } catch (e: any) {
      // Network-level error — bubble up so callers can decide what to do.
      console.warn("reportViolation network error", e?.message ?? e);
      throw e;
    }
  }, [sessionId, getAuthHeaders]);

  const startInterview = useCallback(async (
    jobTitle?: string,
    difficulty?: string,
    techStack?: string,
    existingSessionId?: string, // NEW: Accept existing session ID
    existingQuestionData?: any  // NEW: Accept existing question data
  ) => {
    setError(null);
    setLastFeedback(null);
    setPerformanceMetrics(null);

    // HYDRATION BRANCH: if caller provided sessionId + first question (page-level start)
    if (existingSessionId && existingQuestionData) {
      console.log("Hydrating interview state from existing session:", existingSessionId);
      console.log("🔥 startInterview (hydration): existingQuestionData:", existingQuestionData);

      setSessionId(existingSessionId);

      // Normalize the question data passed from the Page
      const backendId = existingQuestionData.qaId || existingQuestionData.questionId;
      const normalizedQuestion: InterviewQuestion = {
        questionId: backendId || `q_${Date.now()}`,
        questionText: existingQuestionData.question || existingQuestionData.questionText || "",
        type: normalizeType(existingQuestionData.type),
        target_project: existingQuestionData.target_project,
        technology_focus: existingQuestionData.technology_focus,
        expectedAnswerType: existingQuestionData.expected_answer_type || existingQuestionData.expectedAnswerType || "medium",
        difficulty: existingQuestionData.difficulty || "medium",
        ideal_outline: existingQuestionData.ideal_answer_outline || existingQuestionData.ideal_outline || "",
        red_flags: existingQuestionData.red_flags || [],
        confidence: existingQuestionData.confidence,
          is_probe: existingQuestionData.is_probe || false,
  round: existingQuestionData.round || "screening",

      };
console.log("🧭 CURRENT QUESTION TYPE:", currentQuestion?.type);

      // Preserve full backend payload so nested fields (like coding_challenge/test_cases) are available to the UI
      setCurrentQuestion({
        ...normalizedQuestion,
        raw: existingQuestionData,
        coding_challenge: existingQuestionData.coding_challenge || existingQuestionData.challenge || existingQuestionData
      });

      setStage("running");
      setLoading(false);
      return; // EXIT EARLY - Do not fetch again
    }

    if (!token) {
      setError("Please log in to start an interview.");
      return;
    }

    if (!resumeParsed && !resumeFileUrl) {
      setError("Please upload your resume first.");
      return;
    }

    setLoading(true);
    try {
      const payload: any = {
        resume_summary: resumeParsed?.summary ?? "",
        parsed_resume: resumeParsed ?? {},
        retrieved_chunks: [],
        allow_pii: false,
      };
      if (resumeFileUrl) payload.resume_url = resumeFileUrl;

      const res = await fetch(`${API}/interview/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...getAuthHeaders() },
        body: JSON.stringify(payload),
      });
      const body = await res.json();

      if (!res.ok) {
        throw new Error(body?.message || body?.error || JSON.stringify(body) || "Failed to start interview");
      }

      setSessionId(body.sessionId || body.session_id || null);

      // Handle different response formats
      const questionData = body.firstQuestion || body.question || body.data?.question || body.parsed;

      // Debug: show raw question data returned from server
      console.log("🔥 startInterview: raw questionData:", questionData);

      if (questionData) {
        // FIX: Prioritize 'qaId' from backend, fall back to 'questionId'
        const backendId = questionData.qaId || questionData.questionId;

        const normalizedQuestion: InterviewQuestion = {
          questionId: backendId || `q_${Date.now()}`, // Fallback only if backend fails
          questionText: questionData.question || questionData.questionText || "",
          type: normalizeType(questionData.type), 

          target_project: questionData.target_project,
          technology_focus: questionData.technology_focus,
          expectedAnswerType: question_data_or_default(questionData),
          difficulty: questionData.difficulty || "medium",
          ideal_outline: questionData.ideal_answer_outline || questionData.ideal_outline || "",
          red_flags: questionData.red_flags || [],
          confidence: questionData.confidence,
           is_probe: questionData.is_probe || false,
    round: questionData.round || body.round_info?.current || "screening",
        };

        // Preserve raw/coding_challenge so UI can access nested fields like test_cases/starter_code
        setCurrentQuestion({
          ...normalizedQuestion,
          raw: questionData,
          coding_challenge: questionData.coding_challenge || questionData.challenge || questionData
        });
        setStage("running");
      } else {
        setStage("idle");
      }

      setHistory([]);
      setFinalDecision(null);

    } catch (err: any) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  }, [resumeParsed, resumeFileUrl, token, getAuthHeaders]);

  // Helper used above to get expectedAnswerType with safe fallback
  function question_data_or_default(q: any) {
    return q.expected_answer_type || q.expectedAnswerType || "medium";
  }

const submitAnswer = useCallback(
  async (candidateAnswerOrPayload: any, questionIdArg?: string) => {
    setError(null);
    setLastFeedback(null);

    if (!token) { setError("Please log in to submit an answer."); return null; }
    if (!sessionId || !currentQuestion) { setError("Session not initialized or no active question"); return null; }

    setLoading(true);
    try {
      // Normalize incoming arg
      let candidateAnswer: string;
      let code_execution_result = null;
      let question_type = "text";
      
      if (typeof candidateAnswerOrPayload === "string") {
        candidateAnswer = candidateAnswerOrPayload;
      } else {
        // payload object from page
        candidateAnswer = candidateAnswerOrPayload.answer || candidateAnswerOrPayload.candidateAnswer || "";
        code_execution_result = candidateAnswerOrPayload.code_execution_result ?? null;
        question_type = candidateAnswerOrPayload.question_type || "text";
      }

      const conv = buildConversationFromHistory();
      conv.push({ role: "assistant", text: currentQuestion.questionText });
      conv.push({ role: "user", text: candidateAnswer });

      const questionHistory = buildQuestionHistory();

      const payload = {
        sessionId,
        qaId: currentQuestion.questionId,
        questionId: currentQuestion.questionId,
        questionText: currentQuestion.questionText,
        ideal_outline: currentQuestion.ideal_outline || currentQuestion.ideal_answer_outline || "",
        candidateAnswer,
        candidate_answer: candidateAnswer,
        resume_summary: resumeParsed?.summary || "",
        retrieved_chunks: [],
        conversation: conv,
        question_history: questionHistory,
        allow_pii: false,
        question_type, // 👈 NEW
        code_execution_result, // 👈 NEW
      };

      const res = await fetch(`${API}/interview/answer`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...getAuthHeaders() },
        body: JSON.stringify(payload),
      });

      const body = await res.json();
      if (!res.ok) throw new Error(body?.message || body?.error || JSON.stringify(body) || "Answer submit failed");

      // Parse result with multiple fallback paths
      const rawResult = body.result || body.validated || body;

      const result: InterviewAnswerResult = {
        score: rawResult.overall_score || rawResult.score,
        overall_score: rawResult.overall_score || rawResult.score,
        rubricScores: rawResult.rubric_scores || rawResult.rubricScores,
        dimension_scores: rawResult.dimension_scores,
        confidence: rawResult.confidence,
        verdict: rawResult.verdict,
        rationale: rawResult.rationale,
        red_flags_detected: rawResult.red_flags_detected || [],
        missing_elements: rawResult.missing_elements || [],
        improvement: rawResult.improvement || rawResult.follow_up_probe,
        follow_up_probe: rawResult.follow_up_probe,
        ended: body.ended || rawResult.ended || false,
        decision: body.final_decision || body.decision || rawResult.decision,
        in_gray_zone: body.in_gray_zone || rawResult.in_gray_zone || false,
        needs_human_review: body.needs_human_review || rawResult.needs_human_review || false,
        
        // 👇 NEW: Capture round info and elimination
        round_info: body.round_info,
        eliminated: body.eliminated || false,
        elimination_reason: body.elimination_reason || null,
        is_probe: body.nextQuestion?.is_probe || false,
      };

      // Capture feedback/improvement suggestions
      if (result.improvement) {
        setLastFeedback(result.improvement);
      } else if (result.follow_up_probe) {
        setLastFeedback(result.follow_up_probe);
      }

      // Update performance metrics if provided
      if (body.performance_metrics) {
        setPerformanceMetrics(body.performance_metrics);
      }

      // Add to history
      setHistory((h) => [...h, { q: currentQuestion, a: candidateAnswer, result }]);

      // 👇 NEW: Handle elimination immediately
      if (result.eliminated || body.eliminated) {
        console.log("❌ ELIMINATED:", result.elimination_reason || body.elimination_reason);
        
        const eliminationDecision = {
          verdict: "reject",
          confidence: 0.95,
          reason: result.elimination_reason || body.elimination_reason || "Eliminated due to performance",
          eliminated: true,
          performanceMetrics: body.round_info?.round_history || body.performance_metrics,
        };
        
        setFinalDecision(eliminationDecision);
        setCurrentQuestion(null);
        setStage("done");
        return result;
      }

      // Check if interview ended (completion)
      if (result.ended || body.is_final) {
        const decision = result.decision || body.final_decision || body.result?.parsed || body.decision;
        
        console.log("🏁 Interview Ended. Final Decision Data:", decision);
        
        // 👇 NEW: Enhance decision with round history if available
        if (decision && body.round_info?.round_history) {
          decision.performanceMetrics = body.round_info.round_history;
        }
        
        setFinalDecision(decision);
        setCurrentQuestion(null);
        setStage("done");
      } else {
        // Get next question
        const nextQuestionData = body.nextQuestion || body.next_question || body.parsed;

        if (nextQuestionData && nextQuestionData.question !== "Interview Complete") {
          const backendId = nextQuestionData.qaId || nextQuestionData.questionId;

          const normalizedNext: InterviewQuestion = {
            questionId: backendId || `q_${Date.now()}`,
            questionText: nextQuestionData.question || nextQuestionData.questionText ||
              nextQuestionData.followup_question || nextQuestionData.follow_up_question || "",
            type: normalizeType(nextQuestionData.type),
            target_project: nextQuestionData.target_project,
            technology_focus: nextQuestionData.technology_focus,
            expectedAnswerType: nextQuestionData.expected_answer_type || nextQuestionData.expectedAnswerType || "medium",
            difficulty: nextQuestionData.difficulty || "hard",
            ideal_outline: nextQuestionData.ideal_answer_outline || nextQuestionData.ideal_outline || "",
            red_flags: nextQuestionData.red_flags || [],
            confidence: nextQuestionData.confidence,
            
            // 👇 NEW: Preserve probe flag and round info
            is_probe: nextQuestionData.is_probe || false,
            round: nextQuestionData.round || body.round_info?.current || "screening",
          };

          // Preserve raw payload so UI can access nested challenge/testcases
          setCurrentQuestion({
            ...normalizedNext,
            raw: nextQuestionData,
            coding_challenge: nextQuestionData.coding_challenge || nextQuestionData.challenge || nextQuestionData
          });
          setStage("running");
        } else if (result.ended) {
          setStage("done");
          setCurrentQuestion(null);
        } else {
          setCurrentQuestion(null);
          setStage("done");
        }
      }

      return result;

    } catch (err: any) {
      setError(err.message || String(err));
      return null;
    } finally {
      setLoading(false);
    }
  },
  [sessionId, currentQuestion, buildConversationFromHistory, buildQuestionHistory, resumeParsed, token, getAuthHeaders]
);
  const endInterview = useCallback(async (reason?: string, markRejected = false) => {
    setError(null);

    // best-effort: report violation/event before ending if reason present
    if (reason) {
      try {
        await reportViolation(reason);
      } catch (e) {
        console.warn("reportViolation failed before endInterview", e);
      }
    }

    if (!token) {
      setError("Please log in to end the interview.");
      setStage("done");
      setCurrentQuestion(null);
      return;
    }

    if (!sessionId) {
      setStage("done");
      setCurrentQuestion(null);
      return;
    }

    try {
      const res = await fetch(`${API}/interview/end`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...getAuthHeaders() },
        body: JSON.stringify({ sessionId, reason: reason ?? null, terminated_by_violation: !!markRejected }),
      });

      const body = await res.json().catch(() => ({}));
      if (!res.ok) {
        console.warn("endInterview error:", body);
      } else {
        // if backend returned a decision, save it
        if (body && body.finalDecisionRef) {
          setFinalDecision({ ref: body.finalDecisionRef, reason: reason ?? null, terminated_by_violation: !!markRejected });
        } else if (body && body.session && body.session.finalVerdict) {
          setFinalDecision({ verdict: body.session.finalVerdict, reason: body.session.finalReason });
        }
      }
    } catch (e) {
      console.warn("endInterview network error:", e);
    } finally {
      setStage("done");
      setCurrentQuestion(null);
    }
  }, [sessionId, token, getAuthHeaders, reportViolation]);

  return {
    sessionId,
    stage,
    currentQuestion,
    lastFeedback,
    performanceMetrics,
    history,
    loading,
    error,
    resumeParsed,
    resumeFileUrl,
    finalDecision,
    onResumeReady,
    startInterview,
    submitAnswer,
    endInterview,
    reportViolation,      // <-- expose this so components can call it directly if needed
    setResumeParsed,
    setError,
  };
}
