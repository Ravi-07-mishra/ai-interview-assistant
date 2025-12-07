// pages/interview/InterviewPage.tsx
"use client";

import React, { useEffect, useState, useCallback, useRef } from "react";
import ResumeUploader from "../resume/page";
import { useInterview } from "../hooks/useInterview";
import { useAuth } from "../context/AuthContext";
import Link from "next/link";
import {
  Sparkles, X, CheckCircle, AlertCircle, Play,
  TrendingUp, TrendingDown, Minus, Target,
  Award, XCircle, HelpCircle, Lightbulb
} from "lucide-react";

export default function InterviewPage() {
  const {
    stage,
    currentQuestion,
    lastFeedback,
    performanceMetrics,
    history,
    loading,
    error,
    startInterview,
    submitAnswer,
    onResumeReady,
    resumeParsed,
    finalDecision,
    endInterview,
    reportViolation,
    sessionId,
  } = useInterview();

  const { token } = useAuth();
  const [answer, setAnswer] = useState("");
  const [showReport, setShowReport] = useState(false);

  // Violation state (UI)
  const [violationCount, setViolationCount] = useState(0);
  const [showViolationWarning, setShowViolationWarning] = useState(false);
  const [terminatedByViolation, setTerminatedByViolation] = useState(false);
  const [violationReason, setViolationReason] = useState<string | null>(null);

  // Fullscreen enforcement state
  const [fullscreenPromptVisible, setFullscreenPromptVisible] = useState(false); // initial-start prompt
  const [reenterPromptVisible, setReenterPromptVisible] = useState(false); // after warning
  const [needsFullscreen, setNeedsFullscreen] = useState(true);
  const startAttemptRef = useRef(false);

  // Countdown for re-enter modal
  const [countdown, setCountdown] = useState<number>(30);
  const countdownTimerRef = useRef<number | null>(null);

  // Confirmation modal for starting a new interview
  const [confirmRestartVisible, setConfirmRestartVisible] = useState(false);

  // Synchronous refs to avoid races when multiple DOM events fire
  const violationRef = useRef(0);   // immediate counter
  const endingRef = useRef(false);  // prevents duplicate terminations

  useEffect(() => {
    if (resumeParsed) console.log("Resume is ready:", resumeParsed);
  }, [resumeParsed]);

  // Helper: detect fullscreen state (cross-browser)
  function isFullscreenActive() {
    return !!(
      document.fullscreenElement ||
      (document as any).webkitFullscreenElement ||
      (document as any).mozFullScreenElement ||
      (document as any).msFullscreenElement
    );
  }

  // Helper: attempt to request fullscreen (returns true if entered)
  async function tryRequestFullscreen(): Promise<boolean> {
    const el = document.documentElement as any;
    const request =
      el.requestFullscreen?.bind(el) ||
      el.webkitRequestFullscreen?.bind(el) ||
      el.mozRequestFullScreen?.bind(el) ||
      el.msRequestFullscreen?.bind(el);

    if (!request) return false;
    try {
      await request();
      // small delay for the browser to update state
      await new Promise((r) => setTimeout(r, 150));
      return isFullscreenActive();
    } catch (err) {
      console.warn("requestFullscreen failed:", err);
      return false;
    }
  }

  // Modified handleStart: require fullscreen (user gesture) and only then start interview.
  async function handleStart() {
    if (!token) return;
    if (startAttemptRef.current) return;
    startAttemptRef.current = true;

    // reset violation/ref state whenever a fresh interview starts
    violationRef.current = 0;
    endingRef.current = false;
    setViolationCount(0);
    setShowViolationWarning(false);
    setTerminatedByViolation(false);
    setViolationReason(null);

    // If already fullscreen, start immediately
    if (isFullscreenActive()) {
      try {
        await startInterview();
      } finally {
        startAttemptRef.current = false;
      }
      return;
    }

    // Try to request fullscreen (user gesture required)
    const entered = await tryRequestFullscreen();
    if (entered) {
      try {
        await startInterview();
      } finally {
        startAttemptRef.current = false;
      }
      return;
    }

    // Show nicer fullscreen prompt modal (no alert())
    setFullscreenPromptVisible(true);
    startAttemptRef.current = false;
  }

  async function handleSubmitAnswer(e: React.FormEvent) {
    e.preventDefault();
    if (!answer.trim()) return;
    try {
      await submitAnswer(answer.trim());
      setAnswer("");
    } catch (err) {
      console.error("submit failed", err);
    }
  }

  // handleViolation: only report to server (do not call endInterview locally).
  // Use local ref/UI for first-warning UX. Server will atomically decide termination.
  const handleViolation = useCallback(async (reason: string) => {
    // If already terminated locally, ignore further violations
    if (terminatedByViolation) return;

    setViolationReason(reason);

    // best-effort: tell server about violation (server will decide to terminate on >1)
    try {
      await reportViolation?.(reason);
    } catch (e) {
      console.warn("reportViolation error:", e);
    }

    // increment local synchronous counter for UX
    violationRef.current += 1;
    const nowCount = violationRef.current;
    setViolationCount(nowCount);

    if (nowCount === 1) {
      // show a visible warning for the candidate
      setShowViolationWarning(true);
      // show re-enter fullscreen modal with countdown (candidate must act)
      setReenterPromptVisible(true);
      setCountdown(30);
      // start countdown timer
      if (countdownTimerRef.current) {
        window.clearInterval(countdownTimerRef.current);
        countdownTimerRef.current = null;
      }
      countdownTimerRef.current = window.setInterval(() => {
        setCountdown((c) => {
          if (c <= 1) {
            // timer expired -> end interview
            if (countdownTimerRef.current) {
              window.clearInterval(countdownTimerRef.current);
              countdownTimerRef.current = null;
            }
            (async () => {
              try {
                endingRef.current = true;
                setTerminatedByViolation(true);
                await endInterview?.("Failed to re-enter fullscreen after warning", true);
              } catch (e) {
                console.warn("endInterview after countdown error:", e);
              }
            })();
            return 0;
          }
          return c - 1;
        });
      }, 1000);

      // auto-hide the light banner after some seconds (but keep modal)
      window.setTimeout(() => setShowViolationWarning(false), 8000);
      return;
    }

    // on second local detection, show terminated UI and rely on server termination
    if (nowCount >= 2) {
      if (endingRef.current) return;
      endingRef.current = true;
      setTerminatedByViolation(true);
      // best-effort local sync: attempt to end interview (optional)
      try {
        await endInterview?.(`Screen violation: ${reason}`, true);
      } catch (e) {
        console.warn("endInterview attempt after violation (optional):", e);
      } finally {
        // ensure any countdown is cleared
        if (countdownTimerRef.current) {
          window.clearInterval(countdownTimerRef.current);
          countdownTimerRef.current = null;
        }
      }
    }
  }, [reportViolation, endInterview, terminatedByViolation]);

  // Attach event listeners only while interview is running
  useEffect(() => {
    if (stage !== "running") return;

    const onVisibilityChange = () => {
      if (document.hidden) handleViolation("Left tab / window hidden");
    };

    const onFullscreenChange = () => {
      if (!isFullscreenActive()) handleViolation("Exited fullscreen mode");
    };

    const onWindowBlur = () => {
      handleViolation("Window lost focus");
    };

    document.addEventListener("visibilitychange", onVisibilityChange);
    document.addEventListener("fullscreenchange", onFullscreenChange);
    // vendor-prefixed
    document.addEventListener("webkitfullscreenchange", onFullscreenChange as any);
    document.addEventListener("mozfullscreenchange", onFullscreenChange as any);
    window.addEventListener("blur", onWindowBlur);

    return () => {
      document.removeEventListener("visibilitychange", onVisibilityChange);
      document.removeEventListener("fullscreenchange", onFullscreenChange);
      document.removeEventListener("webkitfullscreenchange", onFullscreenChange as any);
      document.removeEventListener("mozfullscreenchange", onFullscreenChange as any);
      window.removeEventListener("blur", onWindowBlur);

      // clear countdown timer if leaving running state
      if (countdownTimerRef.current) {
        window.clearInterval(countdownTimerRef.current);
        countdownTimerRef.current = null;
      }
    };
  }, [stage, handleViolation]);

  // Cleanup countdown on unmount
  useEffect(() => {
    return () => {
      if (countdownTimerRef.current) {
        window.clearInterval(countdownTimerRef.current);
        countdownTimerRef.current = null;
      }
    };
  }, []);

  function renderVerdictBadge(verdict?: string) {
    const v = (verdict || "").toLowerCase();
    const base = "inline-flex items-center px-3 py-1.5 rounded-full text-sm font-bold uppercase tracking-wide";

    if (v === "hire") return (
      <span className={`${base} bg-emerald-100 text-emerald-800 border-2 border-emerald-300`}>
        <Award size={16} className="mr-1.5" /> Hire
      </span>
    );
    if (v === "reject") return (
      <span className={`${base} bg-rose-100 text-rose-800 border-2 border-rose-300`}>
        <XCircle size={16} className="mr-1.5" /> Reject
      </span>
    );
    if (v === "maybe") return (
      <span className={`${base} bg-amber-100 text-amber-800 border-2 border-amber-300`}>
        <HelpCircle size={16} className="mr-1.5" /> Maybe
      </span>
    );

    return <span className={`${base} bg-slate-100 text-slate-800`}>{verdict ?? "Unknown"}</span>;
  }

  function renderScoreBadge(score?: number) {
    if (score === undefined || score === null) return null;

    const percentage = Math.round(score * 100);
    let color = "bg-slate-200 text-slate-700";

    if (score >= 0.85) color = "bg-emerald-100 text-emerald-800 border-emerald-300";
    else if (score >= 0.70) color = "bg-green-100 text-green-800 border-green-300";
    else if (score >= 0.50) color = "bg-amber-100 text-amber-800 border-amber-300";
    else if (score >= 0.30) color = "bg-orange-100 text-orange-800 border-orange-300";
    else color = "bg-rose-100 text-rose-800 border-rose-300";

    return (
      <span className={`inline-flex items-center px-2.5 py-1 rounded-lg text-xs font-bold border-2 ${color}`}>
        {percentage}%
      </span>
    );
  }

  function renderTrendIcon(trend?: string) {
    if (!trend) return null;

    switch (trend) {
      case "improving":
        return <TrendingUp size={18} className="text-green-600" />;
      case "declining":
        return <TrendingDown size={18} className="text-rose-600" />;
      case "stable":
        return <Minus size={18} className="text-blue-600" />;
      default:
        return null;
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 py-10 px-4">
      <div className="max-w-6xl mx-auto">

        {/* Violation banners */}
        {showViolationWarning && !terminatedByViolation && (
          <div className="mb-4 p-4 rounded-xl bg-amber-50 border-2 border-amber-300 text-amber-900 flex items-start gap-3 shadow animate-in fade-in slide-in-from-top-2">
            <AlertCircle size={20} className="shrink-0" />
            <div>
              <div className="font-bold">Warning — Do not change screen</div>
              <div className="text-sm">
                We detected that you switched away from the interview or exited fullscreen: <span className="font-medium">{violationReason}</span>.
                This is a formal warning. You must re-enter fullscreen within 30 seconds or the interview will be terminated.
              </div>
            </div>
          </div>
        )}

        {terminatedByViolation && (
          <div className="mb-4 p-4 rounded-xl bg-rose-50 border-2 border-rose-300 text-rose-900 flex items-start gap-3 shadow animate-in fade-in slide-in-from-top-2">
            <X size={20} className="shrink-0" />
            <div>
              <div className="font-bold">Interview Terminated</div>
              <div className="text-sm">
                The interview was terminated because you changed the screen or exited fullscreen after a previous warning:
                <span className="font-medium"> {violationReason}</span>. Your session has ended. Contact the administrator if you think this was an error.
              </div>
            </div>
          </div>
        )}

        {/* Fullscreen prompt modal (initial start) */}
        {fullscreenPromptVisible && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
            <div className="max-w-lg w-full bg-white rounded-xl p-6 shadow-lg">
              <h3 className="text-xl font-bold mb-2">Enter Full Screen to Begin</h3>
              <p className="mb-4 text-sm text-slate-700 leading-relaxed">
                For exam integrity we require the interview to run in fullscreen. When you enter fullscreen we will lock the interview flow to this window.
                Please click <strong>Enter Fullscreen & Start</strong>. If your browser blocks fullscreen, follow its instructions or press <kbd>F11</kbd>.
                If you prefer not to use fullscreen, you may choose <strong>Start anyway (not recommended)</strong> but this may limit your eligibility.
              </p>

              <div className="flex justify-between items-center gap-3">
                <div className="text-sm text-slate-600">
                  Fullscreen is strongly recommended to protect test integrity.
                </div>

                <div className="flex items-center gap-3">
                  <button
                    className="px-4 py-2 rounded border"
                    onClick={() => {
                      // start anyway fallback (no alert)
                      setFullscreenPromptVisible(false);
                      setNeedsFullscreen(false);
                      startInterview().catch((e) => console.warn("startInterview error:", e));
                    }}
                  >
                    Start anyway (not recommended)
                  </button>

                  <button
                    className="px-4 py-2 rounded bg-indigo-600 text-white"
                    onClick={async () => {
                      setFullscreenPromptVisible(false);
                      const entered = await tryRequestFullscreen();
                      if (entered) {
                        await startInterview().catch((e) => console.warn("startInterview error after fullscreen:", e));
                      } else {
                        // show modal again with explanatory text (no alert)
                        setFullscreenPromptVisible(true);
                      }
                    }}
                  >
                    Enter Fullscreen & Start
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Re-enter fullscreen modal after first warning (strict) */}
        {reenterPromptVisible && !terminatedByViolation && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
            <div className="max-w-xl w-full bg-white rounded-xl p-6 shadow-lg">
              <h3 className="text-xl font-bold mb-2 text-rose-700">Immediate Action Required — Re-enter Full Screen</h3>

              <p className="mb-3 text-sm text-slate-700 leading-relaxed">
                We detected activity that may indicate you left the interview window: <strong>{violationReason}</strong>.
                For the integrity of this assessment you must re-enter fullscreen within the countdown below.
                If you do not re-enter fullscreen within the allotted time the interview will be terminated and flagged.
                Please follow the steps below to re-enter fullscreen, or choose to end the interview now.
              </p>

              <div className="mb-4 flex items-center justify-between gap-4">
                <div className="flex items-center gap-3">
                  <div className="text-3xl font-bold text-rose-600">{countdown}s</div>
                  <div className="text-sm text-slate-600">remaining to re-enter fullscreen</div>
                </div>

                <div className="flex items-center gap-3">
                  <button
                    className="px-4 py-2 rounded border"
                    onClick={async () => {
                      // End interview explicitly (no confirm())
                      try {
                        // clear countdown
                        if (countdownTimerRef.current) {
                          window.clearInterval(countdownTimerRef.current);
                          countdownTimerRef.current = null;
                        }
                        endingRef.current = true;
                        setTerminatedByViolation(true);
                        await endInterview?.("Candidate chose to end interview after warning", true);
                      } catch (e) {
                        console.warn("endInterview error from reenter modal:", e);
                      } finally {
                        setReenterPromptVisible(false);
                      }
                    }}
                  >
                    End Interview
                  </button>

                  <button
                    className="px-4 py-2 rounded bg-indigo-600 text-white"
                    onClick={async () => {
                      // Attempt to re-enter fullscreen
                      const entered = await tryRequestFullscreen();
                      if (entered) {
                        // success: hide modal and clear countdown
                        if (countdownTimerRef.current) {
                          window.clearInterval(countdownTimerRef.current);
                          countdownTimerRef.current = null;
                        }
                        setReenterPromptVisible(false);
                        setShowViolationWarning(false);
                        setCountdown(30);
                        // nothing else to do; server already has the violation record — candidate continued
                      } else {
                        // show small inline explanation that fullscreen failed
                        // keep modal open, but update reason text
                        setViolationReason("Fullscreen blocked or not supported — try pressing F11 or allowing fullscreen in your browser.");
                      }
                    }}
                  >
                    Re-enter Fullscreen Now
                  </button>
                </div>
              </div>

              <div className="text-xs text-slate-500">
                If the button does not work, try pressing <kbd>F11</kbd> (Windows/Linux) or <kbd>Ctrl+Command+F</kbd> (Mac) or allow fullscreen from your browser prompt.
              </div>
            </div>
          </div>
        )}

        {/* Header */}
        <div className="mb-8 flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold flex items-center gap-3 text-slate-900">
              <div className="p-2 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-xl">
                <Sparkles className="text-white" size={28} />
              </div>
              AI Technical Interview
            </h1>
            <p className="text-slate-600 mt-1 ml-14">Deep technical assessment powered by AI</p>
          </div>

          <div className="flex items-center gap-3">
            <div className="text-sm text-slate-600 font-semibold bg-white px-4 py-2 rounded-lg shadow-sm border border-slate-200">
              {stage === "running" ? (
                <span className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
                  Live Interview
                </span>
              ) : stage === "done" ? (
                <span className="flex items-center gap-2">
                  <CheckCircle size={16} className="text-green-600" />
                  Completed
                </span>
              ) : (
                "Ready to Start"
              )}
            </div>
          </div>
        </div>

        {/* Performance Metrics Banner (during interview) */}
        {stage === "running" && performanceMetrics && (
          <div className="mb-6 bg-white rounded-xl shadow-md border border-slate-200 p-5 animate-in fade-in slide-in-from-top-4 duration-500">
            <div className="flex items-center justify-between flex-wrap gap-4">
              <div className="flex items-center gap-2">
                <Target size={20} className="text-indigo-600" />
                <span className="font-bold text-slate-800">Performance Metrics</span>
              </div>

              <div className="flex items-center gap-6 flex-wrap">
                <div className="text-center">
                  <div className="text-2xl font-bold text-slate-900">
                    {performanceMetrics.question_count}
                  </div>
                  <div className="text-xs text-slate-500 uppercase tracking-wide">Questions</div>
                </div>

                <div className="text-center">
                  <div className="text-2xl font-bold text-indigo-600">
                    {Math.round(performanceMetrics.average_score * 100)}%
                  </div>
                  <div className="text-xs text-slate-500 uppercase tracking-wide">Average</div>
                </div>

                {performanceMetrics.last_score !== null && (
                  <div className="text-center">
                    <div className="text-2xl font-bold text-slate-900">
                      {Math.round(performanceMetrics.last_score * 100)}%
                    </div>
                    <div className="text-xs text-slate-500 uppercase tracking-wide">Last Score</div>
                  </div>
                )}

                <div className="flex items-center gap-2">
                  {renderTrendIcon(performanceMetrics.trend)}
                  <div className="text-center">
                    <div className="text-sm font-bold text-slate-900 capitalize">
                      {performanceMetrics.trend.replace('_', ' ')}
                    </div>
                    <div className="text-xs text-slate-500 uppercase tracking-wide">Trend</div>
                  </div>
                </div>

                {(performanceMetrics.consecutive_wins > 0 || performanceMetrics.consecutive_fails > 0) && (
                  <div className="text-center">
                    <div className={`text-xl font-bold ${performanceMetrics.consecutive_wins > 0 ? 'text-green-600' : 'text-rose-600'}`}>
                      {performanceMetrics.consecutive_wins > 0 ? (
                        <>🔥 {performanceMetrics.consecutive_wins}</>
                      ) : (
                        <>⚠️ {performanceMetrics.consecutive_fails}</>
                      )}
                    </div>
                    <div className="text-xs text-slate-500 uppercase tracking-wide">
                      {performanceMetrics.consecutive_wins > 0 ? 'Win Streak' : 'Need Improvement'}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Not Logged In Warning */}
        {!token && (
          <div className="mb-6 p-4 rounded-xl bg-amber-50 border-2 border-amber-200 text-sm text-amber-900 flex items-center gap-3 shadow-sm">
            <AlertCircle size={20} className="shrink-0" />
            <span>
              You must be logged in to upload a resume or run the interview.
              <span className="ml-2 font-bold">
                <Link href="/auth/login" className="underline hover:text-amber-700">Log in</Link> or{" "}
                <Link href="/auth/signup" className="underline hover:text-amber-700">Sign up</Link>.
              </span>
            </span>
          </div>
        )}

        {/* Resume Uploader */}
        {stage !== "running" && stage !== "done" && (
          <div className="mb-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <ResumeUploader onReady={onResumeReady} onStart={handleStart} />
          </div>
        )}

        {/* Start Button */}
        {stage === "idle" && resumeParsed && token && (
          <div className="mb-8 flex justify-center animate-in fade-in zoom-in duration-300">
            <button
              onClick={handleStart}
              disabled={loading}
              className="group relative inline-flex items-center gap-3 px-10 py-5 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-2xl font-bold text-xl shadow-2xl hover:shadow-indigo-300 hover:scale-105 transition-all disabled:opacity-70 disabled:cursor-not-allowed disabled:hover:scale-100"
            >
              {loading ? (
                <>
                  <div className="w-6 h-6 border-3 border-white/30 border-t-white rounded-full animate-spin" />
                  <span>Initializing Interview...</span>
                </>
              ) : (
                <>
                  <span>Begin Technical Interview</span>
                  <div className="bg-white/20 p-2 rounded-full group-hover:translate-x-1 transition-transform">
                    <Play size={20} fill="currentColor" />
                  </div>
                </>
              )}
            </button>
          </div>
        )}

        {/* Error Message */}
        {error && (
          <div className="mb-6 p-4 rounded-xl bg-rose-50 border-2 border-rose-200 text-rose-700 text-sm flex items-center gap-3 shadow-sm animate-in fade-in slide-in-from-top-2 duration-300">
            <X size={20} className="shrink-0" />
            <span className="font-medium">{error}</span>
          </div>
        )}

        {/* ACTIVE INTERVIEW */}
        {stage === "running" && currentQuestion && !terminatedByViolation && (
          <div className="space-y-6 max-w-5xl mx-auto">
            {/* Feedback/Advice Box */}
            {lastFeedback && (
              <div className="p-6 bg-gradient-to-br from-amber-50 via-orange-50 to-yellow-50 border-l-4 border-amber-500 rounded-r-2xl shadow-lg animate-in fade-in slide-in-from-top-4 duration-500">
                <div className="flex items-start gap-4">
                  <div className="p-3 bg-gradient-to-br from-amber-400 to-orange-500 rounded-xl shrink-0 shadow-md">
                    <Lightbulb size={24} className="text-white" />
                  </div>
                  <div className="flex-1">
                    <h4 className="text-sm font-black text-amber-900 uppercase tracking-wider mb-2 flex items-center gap-2">
                      💡 AI Mentor Feedback
                    </h4>
                    <p className="text-amber-900 text-base leading-relaxed font-medium">
                      {lastFeedback}
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Question Card */}
            <div className="bg-white rounded-2xl shadow-2xl border-2 border-slate-200 overflow-hidden">
              {/* Question Header */}
              <div className="bg-gradient-to-r from-indigo-50 to-purple-50 p-6 border-b-2 border-slate-200">
                <div className="flex justify-between items-start mb-3">
                  <span className="text-xs font-black tracking-widest text-indigo-600 uppercase bg-indigo-100 px-3 py-1.5 rounded-lg border-2 border-indigo-200">
                    Question {history.length + 1}
                  </span>

                  <div className="flex items-center gap-2">
                    {currentQuestion.difficulty && (
                      <span className={`text-xs font-bold px-2.5 py-1 rounded-lg ${
                        currentQuestion.difficulty === 'expert' || currentQuestion.difficulty === 'hard'
                          ? 'bg-rose-100 text-rose-700 border-2 border-rose-200'
                          : 'bg-amber-100 text-amber-700 border-2 border-amber-200'
                      }`}>
                        {currentQuestion.difficulty.toUpperCase()}
                      </span>
                    )}
                  </div>
                </div>

                <h2 className="text-2xl font-bold text-slate-900 leading-snug">
                  {currentQuestion.questionText}
                </h2>

                {/* Question Metadata */}
                <div className="mt-4 flex flex-wrap gap-2">
                  {currentQuestion.target_project && (
                    <span className="text-xs bg-blue-100 text-blue-700 px-2.5 py-1 rounded-lg border border-blue-200 font-medium">
                      🎯 {currentQuestion.target_project}
                    </span>
                  )}
                  {currentQuestion.technology_focus && (
                    <span className="text-xs bg-purple-100 text-purple-700 px-2.5 py-1 rounded-lg border border-purple-200 font-medium">
                      ⚡ {currentQuestion.technology_focus}
                    </span>
                  )}
                  {currentQuestion.expectedAnswerType === "code" && (
                    <span className="text-xs bg-green-100 text-green-700 px-2.5 py-1 rounded-lg border border-green-200 font-medium">
                      💻 Code Expected
                    </span>
                  )}
                </div>
              </div>

              {/* Answer Form */}
              <div className="bg-slate-50 p-8 border-t border-slate-200">
                <form onSubmit={handleSubmitAnswer}>
                  {currentQuestion.expectedAnswerType === "code" || currentQuestion.expectedAnswerType === "architectural" ? (
                    <textarea
                      value={answer}
                      onChange={(e) => setAnswer(e.target.value)}
                      placeholder={
                        currentQuestion.expectedAnswerType === "code"
                          ? "// Write your code solution here...\n// Be specific about implementation details"
                          : "Describe the architecture, data flow, and key design decisions..."
                      }
                      rows={14}
                      className="w-full p-5 font-mono text-sm bg-slate-900 text-emerald-400 rounded-xl border-2 border-slate-700 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none resize-y shadow-inner"
                      spellCheck={false}
                    />
                  ) : (
                    <textarea
                      value={answer}
                      onChange={(e) => setAnswer(e.target.value)}
                      placeholder="Type your detailed answer here... Be specific about your implementation and thought process."
                      rows={8}
                      className="w-full p-5 text-base bg-white text-slate-800 rounded-xl border-2 border-slate-300 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none resize-y shadow-sm transition-all"
                    />
                  )}

                  <div className="mt-5 flex items-center justify-between">
                    <button
                      type="button"
                      onClick={() => setAnswer("")}
                      className="text-slate-500 text-sm font-medium hover:text-slate-700 transition-colors px-3 py-2 hover:bg-slate-100 rounded-lg"
                    >
                      Clear Answer
                    </button>

                    <button
                      type="submit"
                      disabled={loading || !token || !answer.trim()}
                      className={`px-8 py-4 rounded-xl font-bold text-lg shadow-xl transition-all transform active:scale-95 flex items-center gap-3 ${
                        !token || !answer.trim() || loading
                          ? "bg-slate-300 text-slate-500 cursor-not-allowed"
                          : "bg-gradient-to-r from-indigo-600 to-purple-600 text-white hover:shadow-indigo-300 hover:scale-105"
                      }`}
                    >
                      {loading ? (
                        <>
                          <div className="w-5 h-5 border-3 border-white/30 border-t-white rounded-full animate-spin"></div>
                          Analyzing Answer...
                        </>
                      ) : (
                        <>
                          Submit Answer
                          <CheckCircle size={20} />
                        </>
                      )}
                    </button>
                  </div>
                </form>
              </div>
            </div>
          </div>
        )}

        {/* FINAL RESULTS */}
        {stage === "done" && (
          <div className="max-w-4xl mx-auto animate-in fade-in zoom-in duration-500">
            <div className="p-10 rounded-3xl bg-white border-2 border-slate-200 shadow-2xl">

              {/* Completion Header */}
              <div className="text-center mb-10">
                <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-gradient-to-br from-green-400 to-emerald-600 text-white mb-5 shadow-lg">
                  <CheckCircle size={40} />
                </div>
                <h2 className="text-4xl font-black text-slate-900 mb-2">Interview Complete</h2>
                <p className="text-slate-600 text-lg">Here's your comprehensive performance analysis</p>
              </div>

              {/* Final Metrics Summary */}
              {performanceMetrics && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                  <div className="text-center p-4 bg-slate-50 rounded-xl border border-slate-200">
                    <div className="text-3xl font-black text-slate-900">{performanceMetrics.question_count}</div>
                    <div className="text-xs text-slate-500 uppercase tracking-wide mt-1">Questions</div>
                  </div>
                  <div className="text-center p-4 bg-indigo-50 rounded-xl border border-indigo-200">
                    <div className="text-3xl font-black text-indigo-600">
                      {Math.round(performanceMetrics.average_score * 100)}%
                    </div>
                    <div className="text-xs text-slate-500 uppercase tracking-wide mt-1">Avg Score</div>
                  </div>
                  <div className="text-center p-4 bg-purple-50 rounded-xl border border-purple-200">
                    <div className="text-3xl font-black text-purple-600 capitalize">
                      {performanceMetrics.trend}
                    </div>
                    <div className="text-xs text-slate-500 uppercase tracking-wide mt-1">Trend</div>
                  </div>
                  <div className="text-center p-4 bg-emerald-50 rounded-xl border border-emerald-200">
                    <div className="text-3xl font-black text-emerald-600">
                      {Math.round(performanceMetrics.confidence * 100)}%
                    </div>
                    <div className="text-xs text-slate-500 uppercase tracking-wide mt-1">Confidence</div>
                  </div>
                </div>
              )}

              {/* Decision Badge */}
              {finalDecision ? (
                <div className="bg-gradient-to-br from-slate-50 to-indigo-50 rounded-2xl p-8 mb-8 text-center border-2 border-slate-200">
                  <div className="text-sm text-slate-600 uppercase tracking-widest font-black mb-4">
                    Final Verdict
                  </div>
                  <div className="mb-5 transform scale-150 inline-block">
                    {renderVerdictBadge(finalDecision.verdict)}
                  </div>

                  {finalDecision.confidence && (
                    <div className="text-sm text-slate-500 mb-5 font-medium">
                      Decision Confidence: <span className="font-bold text-slate-700">
                        {(finalDecision.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  )}

                  {finalDecision.reason && (
                    <div className="text-slate-800 font-medium italic max-w-2xl mx-auto text-lg leading-relaxed bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
                      "{finalDecision.reason}"
                    </div>
                  )}

                  {finalDecision.recommended_role && (
                    <div className="mt-5 text-sm text-indigo-600 font-bold">
                      Recommended Role: {finalDecision.recommended_role}
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center p-8 bg-slate-50 rounded-2xl text-slate-500 mb-8 border-2 border-slate-200">
                  Processing final results...
                </div>
              )}

              {/* Action Buttons */}
              <div className="flex justify-center gap-4">
                <button
                  onClick={() => setShowReport(!showReport)}
                  className="px-6 py-3 bg-white border-2 border-slate-300 text-slate-700 rounded-xl hover:bg-slate-50 font-bold transition-all shadow-md hover:shadow-lg"
                >
                  {showReport ? "Hide Full Transcript" : "View Full Transcript"}
                </button>

                <button
                  onClick={() => setConfirmRestartVisible(true)}
                  className="px-6 py-3 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-xl hover:shadow-xl font-bold transition-all shadow-md"
                >
                  Start New Interview
                </button>
              </div>
            </div>

            {/* Confirm restart inline modal (replaces confirm()) */}
            {confirmRestartVisible && (
              <div className="mt-6 max-w-2xl mx-auto p-6 bg-white rounded-xl border border-slate-200 shadow-md">
                <div className="flex justify-between items-start">
                  <div>
                    <h4 className="font-bold text-lg">Start a new interview?</h4>
                    <p className="text-sm text-slate-600">
                      This will clear current progress and begin a fresh session. Are you sure you want to continue?
                    </p>
                  </div>
                  <div className="flex gap-3">
                    <button
                      className="px-4 py-2 rounded border"
                      onClick={() => setConfirmRestartVisible(false)}
                    >
                      Cancel
                    </button>
                    <button
                      className="px-4 py-2 rounded bg-indigo-600 text-white"
                      onClick={() => {
                        setConfirmRestartVisible(false);
                        // start fresh interview
                        handleStart();
                      }}
                    >
                      Yes, start new
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* Detailed Transcript */}
            {showReport && (
              <div className="mt-10 space-y-5">
                <h3 className="font-black text-2xl text-slate-900 px-2 flex items-center gap-3">
                  <div className="w-1 h-8 bg-indigo-600 rounded-full"></div>
                  Complete Transcript
                </h3>

                {history.map((h, idx) => (
                  <div key={idx} className="bg-white p-6 rounded-2xl border-2 border-slate-200 shadow-md hover:shadow-xl transition-shadow">
                    <div className="flex gap-5">
                      <div className="shrink-0 w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 text-white flex items-center justify-center font-black text-sm shadow-md">
                        Q{idx + 1}
                      </div>
                      <div className="flex-1">
                        <div className="font-bold text-slate-900 mb-3 text-lg">{h.q.questionText}</div>

                        <div className="bg-slate-50 p-4 rounded-xl text-slate-700 text-sm mb-4 border-2 border-slate-100 font-mono">
                          {String(h.a)}
                        </div>

                        {h.result && (
                          <div className="space-y-3">
                            <div className="flex items-center gap-4 flex-wrap">
                              <div className="flex items-center gap-2">
                                <span className="text-xs text-slate-500 font-medium">Overall Score:</span>
                                {renderScoreBadge(h.result.score || h.result.overall_score)}
                              </div>

                              {h.result.verdict && (
                                <div className="flex items-center gap-2">
                                  <span className="text-xs text-slate-500 font-medium">Verdict:</span>
                                  <span className={`text-xs font-bold px-2 py-1 rounded ${
                                    h.result.verdict === 'exceptional' || h.result.verdict === 'strong'
                                      ? 'bg-green-100 text-green-800'
                                      : h.result.verdict === 'acceptable'
                                      ? 'bg-blue-100 text-blue-800'
                                      : h.result.verdict === 'weak'
                                      ? 'bg-amber-100 text-amber-800'
                                      : 'bg-rose-100 text-rose-800'
                                  }`}>
                                    {h.result.verdict.toUpperCase()}
                                  </span>
                                </div>
                              )}
                            </div>

                            {h.result.rationale && (
                              <div className="text-xs text-slate-600 bg-blue-50 p-3 rounded-lg border border-blue-100">
                                <span className="font-bold text-blue-900">Rationale: </span>
                                {h.result.rationale}
                              </div>
                            )}

                            {h.result.red_flags_detected && h.result.red_flags_detected.length > 0 && (
                              <div className="text-xs text-rose-700 bg-rose-50 p-3 rounded-lg border border-rose-200">
                                <span className="font-bold">⚠️ Red Flags: </span>
                                {h.result.red_flags_detected.join(", ")}
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
