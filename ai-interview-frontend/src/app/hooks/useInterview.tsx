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
    result?: InterviewAnswerResult 
  }>>([]);
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [resumeParsed, setResumeParsed] = useState<any | null>(null);
  const [resumeFileUrl, setResumeFileUrl] = useState<string | null>(null);
  const [finalDecision, setFinalDecision] = useState<any | null>(null);

  const onResumeReady = useCallback((parsed: any, fileUrl?: string | null) => {
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
      questionText: h.q.questionText,
      answer: h.a,
      score: h.result?.score || h.result?.overall_score || 0,
      verdict: h.result?.verdict,
      ideal_outline: h.q.ideal_outline || h.q.ideal_answer_outline
    }));
  }, [history]);

  const getAuthHeaders = useCallback(() => {
    if (!token) return {};
    return { Authorization: `Bearer ${token}` };
  }, [token]);

  const startInterview = useCallback(async () => {
    setError(null);
    setLastFeedback(null);
    setPerformanceMetrics(null);

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
      
      if (questionData) {
        // FIX: Prioritize 'qaId' from backend, fall back to 'questionId'
        const backendId = questionData.qaId || questionData.questionId; 

        const normalizedQuestion: InterviewQuestion = {
          questionId: backendId || `q_${Date.now()}`, // Fallback only if backend fails
          questionText: questionData.question || questionData.questionText || "",
          target_project: questionData.target_project,
          technology_focus: questionData.technology_focus,
          expectedAnswerType: questionData.expected_answer_type || questionData.expectedAnswerType || "medium",
          difficulty: questionData.difficulty || "medium",
          ideal_outline: questionData.ideal_answer_outline || questionData.ideal_outline || "",
          red_flags: questionData.red_flags || [],
          confidence: questionData.confidence
        };
        
        setCurrentQuestion(normalizedQuestion);
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

  const submitAnswer = useCallback(
    async (candidateAnswer: string) => {
      setError(null);
      setLastFeedback(null);

      if (!token) {
        setError("Please log in to submit an answer.");
        return null;
      }

      if (!sessionId || !currentQuestion) {
        setError("Session not initialized or no active question");
        return null;
      }
      
      setLoading(true);
      try {
        const conv = buildConversationFromHistory();
        conv.push({ role: "assistant", text: currentQuestion.questionText });
        conv.push({ role: "user", text: candidateAnswer });

        const questionHistory = buildQuestionHistory();

        // FIX: Ensure we send the correct ID to the backend
        const payload = {
          sessionId,
          qaId: currentQuestion.questionId, // This must match the backend's UUID
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
        };

        const res = await fetch(`${API}/interview/answer`, {
          method: "POST",
          headers: { "Content-Type": "application/json", ...getAuthHeaders() },
          body: JSON.stringify(payload),
        });
        const body = await res.json();
        
        if (!res.ok) {
          throw new Error(body?.message || body?.error || JSON.stringify(body) || "Answer submit failed");
        }

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
          decision: body.decision || rawResult.decision,
          in_gray_zone: body.in_gray_zone || rawResult.in_gray_zone || false,
          needs_human_review: body.needs_human_review || rawResult.needs_human_review || false
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

        // Check if interview ended
        if (result.ended || body.is_final) {
          const decision = result.decision || body.result?.parsed || body.decision;
          setFinalDecision(decision);
          setCurrentQuestion(null);
          setStage("done");
        } else {
          // Get next question
          const nextQuestionData = body.nextQuestion || body.next_question || body.parsed;
          
          if (nextQuestionData) {
            // FIX: Prioritize 'qaId' for the next question as well
            const backendId = nextQuestionData.qaId || nextQuestionData.questionId;

            const normalizedNext: InterviewQuestion = {
              questionId: backendId || `q_${Date.now()}`,
              questionText: nextQuestionData.question || nextQuestionData.questionText || 
                            nextQuestionData.followup_question || nextQuestionData.follow_up_question || "",
              target_project: nextQuestionData.target_project,
              technology_focus: nextQuestionData.technology_focus,
              expectedAnswerType: nextQuestionData.expected_answer_type || nextQuestionData.expectedAnswerType || "medium",
              difficulty: nextQuestionData.difficulty || "hard",
              ideal_outline: nextQuestionData.ideal_answer_outline || nextQuestionData.ideal_outline || "",
              red_flags: nextQuestionData.red_flags || [],
              confidence: nextQuestionData.confidence
            };
            
            setCurrentQuestion(normalizedNext);
            setStage("running");
          } else {
            // No next question but not ended - might need to request one
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

  const endInterview = useCallback(async () => {
    setError(null);
    
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
        body: JSON.stringify({ sessionId }),
      });
      
      if (!res.ok) {
        const b = await res.json().catch(() => ({}));
        console.warn("endInterview error:", b);
      }
    } catch (e) {
      console.warn("endInterview network error:", e);
    } finally {
      setStage("done");
      setCurrentQuestion(null);
    }
  }, [sessionId, token, getAuthHeaders]);

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
    setResumeParsed,
    setError,
  };
}