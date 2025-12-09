"use client";

import React, { useEffect, useState, useCallback, useRef } from "react";
import ResumeUploader from "../resume/page";
import { useInterview } from "../hooks/useInterview";
import { useAuth } from "../context/AuthContext";
import Link from "next/link";
import {
  Sparkles,
  X,
  CheckCircle,
  AlertCircle,
  Play,
  TrendingUp,
  TrendingDown,
  Minus,
  Target,
  Award,
  XCircle,
  HelpCircle,
  Lightbulb,
  Loader2, // Added for loading indicator
} from "lucide-react";

/* -------------------------
    Helper render functions (unchanged)
    ------------------------- */
const renderScoreBadge = (score: number | undefined) => {
  if (score === undefined || score === null)
    return (
      <span className="text-xs font-bold px-2 py-1 rounded bg-slate-100 text-slate-800">
        N/A
      </span>
    );
  const percent = Math.round(score * 100);
  let color = "bg-rose-100 text-rose-800";
  if (percent >= 75) color = "bg-green-100 text-green-800";
  else if (percent >= 50) color = "bg-blue-100 text-blue-800";
  else if (percent >= 25) color = "bg-amber-100 text-amber-800";

  return (
    <span className={`text-sm font-black px-3 py-1 rounded-full ${color}`}>
      {percent}%
    </span>
  );
};

const renderVerdictBadge = (verdict: string | undefined) => {
  if (verdict === undefined || verdict === null) {
    return (
      <span
        style={{
          fontSize: "1.5rem",
          padding: "0.5rem 1.5rem",
          borderRadius: "9999px",
          fontWeight: 900,
          textShadow: "0 1px 3px rgba(0,0,0,0.1)",
        }}
        className="bg-slate-200 text-slate-700"
      >
        PENDING
      </span>
    );
  }

  let style: React.CSSProperties = {
    fontSize: "1.5rem",
    padding: "0.5rem 1.5rem",
    borderRadius: "9999px",
    fontWeight: 900,
    textShadow: "0 1px 3px rgba(0,0,0,0.1)",
  };

  switch (verdict.toLowerCase()) {
    case "strong hire":
    case "exceptional":
      style = {
        ...style,
        background:
          "linear-gradient(to right, var(--tw-color-emerald-500), var(--tw-color-green-700))",
        color: "white",
      };
      return (
        <span style={style} className="shadow-lg shadow-emerald-200">
          <Award size={24} className="inline mr-2" /> STRONG HIRE
        </span>
      );
    case "acceptable":
      style = {
        ...style,
        background:
          "linear-gradient(to right, var(--tw-color-blue-500), var(--tw-color-indigo-700))",
        color: "white",
      };
      return (
        <span style={style} className="shadow-lg shadow-indigo-200">
          <CheckCircle size={24} className="inline mr-2" /> ACCEPTABLE
        </span>
      );
    case "weak":
    case "fail":
      style = {
        ...style,
        background:
          "linear-gradient(to right, var(--tw-color-rose-500), var(--tw-color-red-700))",
        color: "white",
      };
      return (
        <span style={style} className="shadow-lg shadow-rose-200">
          <XCircle size={24} className="inline mr-2" /> NOT RECOMMENDED
        </span>
      );
    default:
      return (
        <span style={style} className="bg-slate-200 text-slate-700">
          {verdict.toUpperCase()}
        </span>
      );
  }
};

const renderTrendIcon = (trend: string) => {
  switch (trend) {
    case "increasing":
      return <TrendingUp size={24} className="text-green-600" />;
    case "decreasing":
    case "falling":
      return <TrendingDown size={24} className="text-rose-600" />;
    case "stable":
    default:
      return <Minus size={24} className="text-slate-600" />;
  }
};

/* -------------------------
    Main component
    ------------------------- */

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

  // Camera refs/state: separate preview and proctor video elements
  const previewVideoRef = useRef<HTMLVideoElement | null>(null);
  const proctorVideoRef = useRef<HTMLVideoElement | null>(null);
  const previewCanvasRef = useRef<HTMLCanvasElement | null>(null); // for reference capture
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const inFlightRef = useRef(false);

  const [cameraActive, setCameraActive] = useState(false);
  // referenceImage is now the source of truth for successful client-side capture
  const [referenceImage, setReferenceImage] = useState<string | null>(null);

  // 📸 NEW: Explicit status for client-side image capture and validation
  const [imageStatus, setImageStatus] = useState<"pending" | "capturing" | "captured" | "error">("pending");

  const [cameraError, setCameraError] = useState<string | null>(null);
  const [cameraPermissionRequested, setCameraPermissionRequested] = useState(false);


  // Prefer a single API base env var (fallbacks supported)
  const API =
    process.env.NEXT_PUBLIC_API_URL || process.env.NEXT_PUBLIC_AI_URL || "";

  // Fullscreen enforcement state
  const [fullscreenPromptVisible, setFullscreenPromptVisible] = useState(false);
  const [reenterPromptVisible, setReenterPromptVisible] = useState(false);
  const [needsFullscreen, setNeedsFullscreen] = useState(true);
  const startAttemptRef = useRef(false); // Used to prevent duplicate handleStart calls

  // Countdown for re-enter modal
  const [countdown, setCountdown] = useState<number>(30);
  const countdownTimerRef = useRef<number | null>(null);

  // Confirmation modal for starting a new interview
  const [confirmRestartVisible, setConfirmRestartVisible] = useState(false);

  // Synchronous refs to avoid races when multiple DOM events fire
  const violationRef = useRef(0); // immediate counter
  const endingRef = useRef(false); // prevents duplicate terminations

  /* -------------------------
      Helper: stop camera stream
      ------------------------- */
  const stopCamera = useCallback(() => {
    try {
      // stop both video elements if they have streams
      [previewVideoRef.current, proctorVideoRef.current].forEach((videoEl) => {
        if (videoEl && videoEl.srcObject) {
          const stream = videoEl.srcObject as MediaStream;
          stream.getTracks().forEach((track) => track.stop());
          videoEl.srcObject = null;
        }
      });
      setCameraActive(false);
      // NOTE: Do NOT clear referenceImage/imageStatus here if the user is just stopping preview.
      // We only clear it on error or successful endInterview.
      console.log("Camera streams stopped.");
    } catch (e) {
      console.warn("stopCamera error:", e);
    }
  }, []);

  /* -------------------------
      Violation wrapper (unchanged behavior)
      ------------------------- */
const VIOLATION_THRESHOLD = 2;

const reportViolationWrapper = useCallback(
  async (reason: string, isTerminal: boolean = false) => {
    // prevent doing anything if we've already ended
    if (endingRef.current) return;

    // optimistic increment of the client-side counter (still update from server next)
    violationRef.current += 1;
    setViolationCount(violationRef.current);
    setViolationReason(reason);

    // compute intended action to send to server (server ultimately decides)
    const intendedAction: "warning" | "terminate" =
      isTerminal || violationRef.current >= VIOLATION_THRESHOLD ? "terminate" : "warning";

    console.warn(
      `[VIOLATION - client] reason=${reason} localCount=${violationRef.current} sendAction=${intendedAction}`
    );

    let serverResp: any = null;
    try {
      serverResp = await reportViolation(reason, intendedAction, sessionId);
      console.info("[VIOLATION] serverResp:", serverResp);
    } catch (err) {
      console.error("Error reporting violation to server:", err);
    }

    // reconcile local counter with server (if available)
    const serverCount =
      serverResp && typeof serverResp.violationCount === "number"
        ? serverResp.violationCount
        : null;

    if (serverCount !== null) {
      violationRef.current = serverCount;
      setViolationCount(serverCount);
    }

    const serverTerminated = !!(serverResp && serverResp.terminated);

    // Decide whether to terminate locally:
    const shouldTerminateLocally =
      serverTerminated ||
      (serverCount !== null ? serverCount >= VIOLATION_THRESHOLD : intendedAction === "terminate");

    if (shouldTerminateLocally) {
      if (endingRef.current) return;
      endingRef.current = true;

      if (countdownTimerRef.current) {
        window.clearInterval(countdownTimerRef.current);
        countdownTimerRef.current = null;
      }

      setTerminatedByViolation(true);
      setReenterPromptVisible(false);
      stopCamera();

      try {
        console.warn(`[VIOLATION] terminating interview (reason=${reason})`);
        const endReason =
          serverResp?.message ||
          serverResp?.endedReason ||
          `Interview terminated due to multiple integrity violations: ${reason}`;

        await endInterview?.(endReason, true);
      } catch (e) {
        console.error("Error ending interview after termination:", e);
      }

      return;
    }

    // If not terminating: show re-enter UI and start countdown
    setShowViolationWarning(true);
    setReenterPromptVisible(true);
    setCountdown(30);

    if (countdownTimerRef.current) {
      window.clearInterval(countdownTimerRef.current);
      countdownTimerRef.current = null;
    }

    const localCountSnapshot = violationRef.current;
    countdownTimerRef.current = window.setInterval(() => {
      setCountdown((prev) => {
        if (prev <= 1) {
          if (countdownTimerRef.current) {
            window.clearInterval(countdownTimerRef.current);
            countdownTimerRef.current = null;
          }
          (async () => {
            try {
              if (!endingRef.current) {
                await reportViolationWrapper(
                  `Fullscreen not re-entered within 30 seconds (Violation Count: ${localCountSnapshot + 1})`,
                  true
                );
              }
            } catch (e) {
              console.error("Error auto-escalating violation on countdown end:", e);
            }
          })();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
  },
  [reportViolation, sessionId, endInterview, stopCamera, showViolationWarning]
);

  useEffect(() => {
    if (resumeParsed) console.log("Resume is ready:", resumeParsed);
  }, [resumeParsed]);

  /* --------------------------------------------------------------------------
      Reference capture (client-side quality checks only)
      - Sets referenceImage state on client-side success
      -------------------------------------------------------------------------- */
const MIN_IMAGE_LENGTH = 10000;

const captureReferenceImage = useCallback(async () => {
  setImageStatus("capturing");
  setCameraError(null);
  
  const MAX_RETRIES = 3;
  let lastError = null;
  
  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    try {
      const videoEl = previewVideoRef.current;
      const canvas = previewCanvasRef.current;

      if (!videoEl || !canvas) {
        throw new Error("Video/canvas not available");
      }

      // Ensure fresh stream
      if (!videoEl.srcObject || attempt > 1) {
        const existingStream = videoEl.srcObject as MediaStream;
        if (existingStream) {
          existingStream.getTracks().forEach(t => t.stop());
        }
        
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            facingMode: "user"
          },
          audio: false,
        });
        
        videoEl.srcObject = stream;
        videoEl.muted = true;
        videoEl.playsInline = true;
      }

      // Wait for ready state
      const maxWaitMs = 5000;
      const pollInterval = 100;
      let waited = 0;
      
      while ((videoEl.readyState || 0) < 2 && waited < maxWaitMs) {
        await new Promise((r) => setTimeout(r, pollInterval));
        waited += pollInterval;
      }

      try {
        await videoEl.play();
      } catch (playErr) {
        console.warn("Autoplay blocked:", playErr);
      }

      // CRITICAL: Wait longer for stable frame
      await new Promise((r) => setTimeout(r, 1000));

      if (!videoEl.videoWidth || !videoEl.videoHeight) {
        throw new Error(`No video frames (attempt ${attempt}/${MAX_RETRIES})`);
      }

      const ctx = canvas.getContext("2d");
      canvas.width = videoEl.videoWidth;
      canvas.height = videoEl.videoHeight;
      
      if (!ctx) throw new Error("Canvas context unavailable");
      
      ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);

      // Get image
      const imageDataUrl = canvas.toDataURL("image/jpeg", 0.90); // Higher quality
      
      if (!imageDataUrl || imageDataUrl.length < 10000) {
        throw new Error(`Image too small: ${imageDataUrl?.length} bytes`);
      }

      // Analyze quality
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imageData.data;
      
      let sum = 0;
      const sampleLimit = Math.min(data.length, 50000);
      
      for (let i = 0; i < sampleLimit; i += 4) {
        const r = data[i], g = data[i + 1], b = data[i + 2];
        sum += (0.299 * r + 0.587 * g + 0.114 * b);
      }
      
      const averageBrightness = sum / (sampleLimit / 4);
      
      // RELAXED thresholds
      if (averageBrightness < 25) {
        throw new Error(`Too dark (${averageBrightness.toFixed(1)}). Turn on lights and retry.`);
      }
      if (averageBrightness > 240) {
        throw new Error(`Too bright (${averageBrightness.toFixed(1)}). Reduce backlight.`);
      }

      // Variance check
      let mean = 0;
      const luminances: number[] = [];
      
      for (let i = 0; i < sampleLimit; i += 4) {
        const lum = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
        luminances.push(lum);
        mean += lum;
      }
      
      mean = mean / luminances.length;
      let variance = 0;
      
      for (const lum of luminances) {
        variance += (lum - mean) ** 2;
      }
      
      variance = variance / luminances.length;

      // RELAXED variance threshold
      if (variance < 150) {
        throw new Error(`Low contrast (${variance.toFixed(1)}). Ensure clear face visibility.`);
      }

      // Basic face detection heuristic (fallback if API unavailable)
      let faceDetected = false;
      
      try {
        const FaceDetector = (window as any).FaceDetector;
        if (typeof FaceDetector === "function") {
          const detector = new FaceDetector();
          const faces = await detector.detect(canvas as any);
          faceDetected = !!(faces && faces.length > 0);
        } else {
          // Heuristic: check for skin-like pixels in center
          const cx = Math.floor(canvas.width / 2);
          const cy = Math.floor(canvas.height / 2);
          const boxW = Math.floor(canvas.width * 0.4);
          const boxH = Math.floor(canvas.height * 0.4);
          
          let skinLike = 0, samples = 0;
          const sx = Math.max(0, cx - Math.floor(boxW / 2));
          const sy = Math.max(0, cy - Math.floor(boxH / 2));
          
          for (let y = sy; y < sy + boxH; y += 8) {
            for (let x = sx; x < sx + boxW; x += 8) {
              const idx = (y * canvas.width + x) * 4;
              const r = data[idx], g = data[idx + 1], b = data[idx + 2];
              
              if (r > 95 && g > 40 && b > 20 && r > g && r > b) {
                skinLike++;
              }
              samples++;
            }
          }
          
          faceDetected = samples > 0 && (skinLike / samples) > 0.06; // Relaxed from 0.08
        }
      } catch (detErr) {
        console.warn("Face detection attempt failed:", detErr);
        // DON'T fail here - let backend handle it
        faceDetected = true;
      }

      if (!faceDetected) {
        throw new Error("No face detected. Center your face and try again.");
      }

      // SUCCESS
      setCameraActive(true);
      setReferenceImage(imageDataUrl);
      setImageStatus("captured");
      
      console.info(`✅ Reference image captured (attempt ${attempt}): ${imageDataUrl.length} bytes`);
      return imageDataUrl;

    } catch (err: any) {
      lastError = err;
      console.warn(`Capture attempt ${attempt}/${MAX_RETRIES} failed:`, err.message);
      
      if (attempt < MAX_RETRIES) {
        await new Promise(r => setTimeout(r, 500)); // Brief pause before retry
      }
    }
  }

  // All retries failed
  console.error("All capture attempts failed:", lastError);
  
  setCameraActive(false);
  setReferenceImage(null);
  
  const errorMessage = lastError?.message || "Camera capture failed after multiple attempts";
  setCameraError(errorMessage);
  setImageStatus("error");
  
  throw lastError;
}, []);


  /* -------------------------
      Proctor capture (uses proctorVideoRef & captureCanvasRef)
      - returns dataURL or null (unchanged)
      ------------------------- */
const captureFrameToDataUrl = useCallback(async (): Promise<string | null> => {
  const video = proctorVideoRef.current;
  const canvas = captureCanvasRef.current;
  if (!video || !canvas) return null;

  // Wait for a frame
  const maxWait = 1500;
  const step = 100;
  let waited = 0;
  while ((video.readyState || 0) < 2 && waited < maxWait) {
    await new Promise((r) => setTimeout(r, step));
    waited += step;
  }

  if (!video.videoWidth || !video.videoHeight) {
    console.debug("captureFrameToDataUrl: video not ready (width/height = 0)");
    return null;
  }

const w = video.videoWidth; 
  const h = video.videoHeight;
  canvas.width = w;
  canvas.height = h;

  const ctx = canvas.getContext("2d");
  if (!ctx) return null;

  try {
    ctx.save();
    ctx.drawImage(video, 0, 0, w, h);
    ctx.restore();

    const dataUrl = canvas.toDataURL("image/jpeg", 0.95);
    if (!dataUrl || typeof dataUrl !== "string") return null;
    return dataUrl;
  } catch (err) {
    console.warn("captureFrameToDataUrl error:", err);
    return null;
  }
}, []);


  /* -------------------------
      Proctoring effect (interval + warmup) (unchanged)
      ------------------------- */
  useEffect(() => {
    let proctorInterval: number | null = null;

    if (stage !== "running" || !cameraActive || !token) {
      return () => {};
    }

    // NOTE: This logic should ideally rely on the sessionId being populated from the startInterview call.

    const isValidFrame = (dataUrl: string | null): boolean => {
      if (!dataUrl || typeof dataUrl !== "string") return false;
      if (dataUrl.length < 500) {
        console.warn(`[Frame Validation] Too short (${dataUrl?.length}) — rejecting.`);
        return false;
      }
      if (!dataUrl.startsWith("data:image/")) {
        console.warn(`[Frame Validation] Missing data:image prefix. Start: ${String(dataUrl).substring(0,36)}`);
        return false;
      }
      const re = /^data:image\/(png|jpeg|jpg|webp);base64,[A-Za-z0-9+/]+=*$/i;
      if (!re.test(dataUrl)) {
        console.warn(`[Frame Validation] Regex mismatch. Start: ${String(dataUrl).substring(0,64)}`);
        return false;
      }
      return true;
    };

    const sendProctorPayload = async (payload: Record<string, any>) => {
      try {
        if (payload.image) {
          console.debug("[proctor -> server] sending image sample:", String(payload.image).substring(0, 80), "len=", String(payload.image).length);
        } else if (payload.status === "no_face") {
          console.debug("[proctor -> server] sending NO_FACE:", { sessionId: payload.sessionId, sample: payload.sample });
        }
        const res = await fetch(`${API || ""}/interview/proctor`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify(payload),
        });
        const j = await res.json().catch(() => null);
       const hasError = !res.ok || j?.verified === false || j?.status === "failed";

        if (hasError) {
          // Extract the actual message from various possible keys
          const violationReason = j?.error || j?.reason || j?.detail || "Face verification failed";
          
          console.warn(`[PROCTOR VIOLATION detected] Reason: ${violationReason}`);
          
          // Trigger the red warning banner
          reportViolationWrapper(violationReason, false);
        } 
        else if (j?.status === "success" || j?.verified === true) {
          // Clear warning if passing
          if (showViolationWarning) setShowViolationWarning(false);
        }
        // --- FIX END ---
        return { ok: res.ok, statusCode: res.status, body: j };
      } catch (err) {
        console.warn("proctor POST failed:", err);
        return { ok: false, error: err };
      }
    };

    const warmupAndStart = async () => {
      if (!sessionId) {
        console.debug("proctor warmup: sessionId missing, skipping warmup POST. Waiting for startInterview to update state.");
        return;
      }

      try {
        if (proctorVideoRef.current && !proctorVideoRef.current.srcObject) {
          try {
        const stream = await navigator.mediaDevices.getUserMedia({
  video: { 
    width: { ideal: 1280 }, 
    height: { ideal: 720 }, 
    facingMode: "user" 
  },
  audio: false,
});
            proctorVideoRef.current.srcObject = stream;
            proctorVideoRef.current.muted = true;
            proctorVideoRef.current.playsInline = true;
            try { await proctorVideoRef.current.play(); } catch (e) { console.warn("proctor warmup: play() blocked:", e); }
          } catch (gErr) {
            console.warn("proctor warmup: failed to get userMedia for proctor video:", gErr);
          }
        }

        const firstFrame = await captureFrameToDataUrl();

        if (!isValidFrame(firstFrame)) {
          inFlightRef.current = true;
          try {
            await sendProctorPayload({
              sessionId,
              status: "no_face",
              sample: typeof firstFrame === "string" ? String(firstFrame).substring(0, 60) : null,
              timestamp: new Date().toISOString()
            });
          } finally {
            inFlightRef.current = false;
          }
          return;
        }

        inFlightRef.current = true;
        try {
          await sendProctorPayload({ sessionId, image: firstFrame });
        } finally {
          inFlightRef.current = false;
        }
      } catch (e) {
        console.warn("proctor warmup error:", e);
      }

      // interval checks
      proctorInterval = window.setInterval(async () => {
        if (!sessionId) { console.debug("proctor interval: sessionId missing, skipping POST"); return; }
        if (inFlightRef.current) return;
        inFlightRef.current = true;
        try {
          const frame = await captureFrameToDataUrl();

          if (!isValidFrame(frame)) {
            try {
              await sendProctorPayload({
                sessionId,
                status: "no_face",
                sample: typeof frame === "string" ? String(frame).substring(0, 60) : null,
                timestamp: new Date().toISOString()
              });
            } catch (err) {
              console.warn("proctor no_face POST failed:", err);
            } finally {
              inFlightRef.current = false;
            }
            return;
          }

          try {
            await sendProctorPayload({ sessionId, image: frame });
          } catch (err) {
            console.warn("proctor image POST failed:", err);
          }
        } catch (err) {
          console.warn("proctor interval error:", err);
        } finally {
          inFlightRef.current = false;
        }
      }, 15000);
    };

    warmupAndStart();

    return () => {
      if (proctorInterval) window.clearInterval(proctorInterval);
    };
  }, [stage, cameraActive, sessionId, token, API, captureFrameToDataUrl, reportViolationWrapper, showViolationWarning]);


  /* -------------------------
      Fullscreen helpers (unchanged)
      ------------------------- */
  const isFullscreen = useCallback((): boolean => {
    return (
      !!document.fullscreenElement ||
      !!(document as any).webkitFullscreenElement ||
      !!(document as any).mozFullScreenElement ||
      !!(document as any).msFullscreenElement
    );
  }, []);

  const tryRequestFullscreen = useCallback(async (): Promise<boolean> => {
    const element = document.documentElement;
    try {
      if (element.requestFullscreen) {
        await element.requestFullscreen();
      } else if ((element as any).mozRequestFullScreen) {
        await (element as any).mozRequestFullScreen();
      } else if ((element as any).webkitRequestFullscreen) {
        await (element as any).webkitRequestFullscreen();
      } else if ((element as any).msRequestFullscreen) {
        await (element as any).msRequestFullscreen();
      }
      return isFullscreen();
    } catch (e) {
      console.error("Fullscreen request failed:", e);
      return false;
    }
  }, [isFullscreen]);

  /* --------------------------------------------------------------------------
      Updated handleStart: Captures Image, then immediately calls /interview/start
      - Guarantees capture and check happens ONLY ONCE on the start click.
      - **CRITICAL:** Only proceeds to the backend call if `referenceImage` is available.
      -------------------------------------------------------------------------- */
const handleStart = useCallback(
  async (
    firstArg: string | object | React.MouseEvent | undefined = "Technical Interview",
    difficulty: string = "medium",
    techStack: string = ""
  ) => {
    const jobTitle = typeof firstArg === 'string' ? firstArg : "Technical Interview";
    
    if (!token) return;
    if (startAttemptRef.current) {
      console.warn("Start already in progress");
      return;
    }
    
    startAttemptRef.current = true;
    setCameraError(null);

    let capturedImage: string | null = referenceImage;
    let serverSessionId: string | null = null;

    try {
      // STEP 1: Ensure image is captured
      if (!capturedImage || imageStatus !== "captured") {
        console.log("Capturing reference image...");
        capturedImage = await captureReferenceImage();
      }

      if (!capturedImage) {
        throw new Error("Reference image capture failed unexpectedly");
      }

      // STEP 2: Fullscreen check
      if (needsFullscreen && !isFullscreen()) {
        setFullscreenPromptVisible(true);
        startAttemptRef.current = false;
        return;
      }
      
      setFullscreenPromptVisible(false);

      // STEP 3: Call backend (single call with registration)
      const startPayload: any = {
        jobTitle,
        difficulty,
        techStack,
        resume_summary: resumeParsed?.summary || "",
        allow_pii: false,
        referenceImage: capturedImage,
      };

      const startUrl = `${API || ""}/interview/start`;
      console.log("🚀 Starting interview with validated image...");

      const resp = await fetch(startUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(startPayload),
      });

      if (resp.status === 200) {
        const data = await resp.json().catch(() => null);
        serverSessionId = data?.sessionId || data?.session_id || null;

        if (!serverSessionId) {
          throw new Error("No session ID returned");
        }

        console.log("✅ Session created:", serverSessionId);

        await startInterview?.(
          jobTitle,
          difficulty,
          techStack,
          serverSessionId,
          data?.firstQuestion
        );
        
        setCameraError(null);

        // Start proctor video
        try {
          if (proctorVideoRef.current && !proctorVideoRef.current.srcObject) {
            const pStream = await navigator.mediaDevices.getUserMedia({
              video: { width: 640, height: 360, facingMode: "user" },
              audio: false,
            });
            
            proctorVideoRef.current.srcObject = pStream;
            proctorVideoRef.current.muted = true;
            proctorVideoRef.current.playsInline = true;
            await proctorVideoRef.current.play().catch(() => {});
          }
        } catch (e) {
          console.warn("Proctor video start failed:", e);
        }

        setCameraActive(true);
        
      } else {
        // Backend returned error
        let body;
        try {
          body = await resp.json();
        } catch (e) {
          body = { message: `Server error ${resp.status}` };
        }

        const errorMessage = body.message || body.error || `Failed to start (status ${resp.status})`;
        throw new Error(errorMessage);
      }

    } catch (e: any) {
      console.error("❌ Start error:", e);

      const displayError = e.message || String(e);
      
      // Provide helpful suggestions based on error
      let suggestion = "";
      if (displayError.includes("dark")) {
        suggestion = " Try turning on more lights.";
      } else if (displayError.includes("bright")) {
        suggestion = " Try reducing backlight or moving away from windows.";
      } else if (displayError.includes("face")) {
        suggestion = " Ensure your face is centered and clearly visible.";
      } else if (displayError.includes("decode") || displayError.includes("format")) {
        suggestion = " Camera issue detected. Try refreshing the page.";
      }

      setCameraError(displayError + suggestion);
      setImageStatus("error");
      stopCamera();
      
    } finally {
      startAttemptRef.current = false;
    }
  },
  [token, needsFullscreen, isFullscreen, startInterview, captureReferenceImage, 
   referenceImage, resumeParsed, API, stopCamera, imageStatus]
);

  /* -------------------------
      Answer submit handler (unchanged)
      ------------------------- */
  const handleSubmitAnswer = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      if (!answer.trim() || loading || !currentQuestion) return;

      const answerToSubmit = answer;
      setAnswer("");
      const currentQId = currentQuestion.id;

      try {
        await submitAnswer(answerToSubmit, currentQId);
      } catch (e) {
        console.error("Error submitting answer:", e);
      }
    },
    [answer, loading, currentQuestion, submitAnswer]
  );

  /* -------------------------
      Cleanup effect (unmount) (unchanged)
      ------------------------- */
  useEffect(() => {
    return () => {
      stopCamera();
      if (countdownTimerRef.current) {
        window.clearInterval(countdownTimerRef.current);
        countdownTimerRef.current = null;
      }
    };
  }, [stopCamera]);

  /* -------------------------
      Fullscreen and Window Event Handlers (unchanged)
      ------------------------- */
  const handleBeforeUnload = useCallback(
    (event: BeforeUnloadEvent) => {
      if (stage === "running" && !endingRef.current) {
        event.preventDefault();
        event.returnValue = "";
        reportViolationWrapper("Attempted page refresh or closing tab.", true);
      }
    },
    [stage, reportViolationWrapper]
  );

  const handleVisibilityChange = useCallback(() => {
    if (stage === "running" && document.visibilityState === "hidden") {
      reportViolationWrapper("Switched to another tab or minimized window.");
    }
    if (
      stage === "running" &&
      document.visibilityState === "visible" &&
      reenterPromptVisible &&
      needsFullscreen &&
      !isFullscreen()
    ) {
      tryRequestFullscreen();
    }
  }, [
    stage,
    reportViolationWrapper,
    reenterPromptVisible,
    needsFullscreen,
    tryRequestFullscreen,
    isFullscreen,
  ]);

  const handleFullscreenChange = useCallback(() => {
    if (stage !== "running" || !needsFullscreen) return;

    if (!isFullscreen() && !reenterPromptVisible && !endingRef.current) {
      reportViolationWrapper("Exited fullscreen mode.");
    } else if (isFullscreen() && reenterPromptVisible) {
      if (countdownTimerRef.current) {
        window.clearInterval(countdownTimerRef.current);
        countdownTimerRef.current = null;
      }
      setReenterPromptVisible(false);
      setShowViolationWarning(false);
      setCountdown(30);
    }
  }, [
    stage,
    needsFullscreen,
    isFullscreen,
    reenterPromptVisible,
    reportViolationWrapper,
  ]);

  useEffect(() => {
    if (stage === "running" && needsFullscreen) {
      window.addEventListener("beforeunload", handleBeforeUnload);
      document.addEventListener("visibilitychange", handleVisibilityChange);
      document.addEventListener("fullscreenchange", handleFullscreenChange);
      document.addEventListener("webkitfullscreenchange", handleFullscreenChange);
      document.addEventListener("mozfullscreenchange", handleFullscreenChange);
      document.addEventListener("msfullscreenchange", handleFullscreenChange);
    }

    return () => {
      window.removeEventListener("beforeunload", handleBeforeUnload);
      document.removeEventListener("visibilitychange", handleVisibilityChange);
      document.removeEventListener("fullscreenchange", handleFullscreenChange);
      document.removeEventListener("webkitfullscreenchange", handleFullscreenChange);
      document.removeEventListener("mozfullscreenchange", handleFullscreenChange);
      document.removeEventListener("msfullscreenchange", handleFullscreenChange);
    };
  }, [
    stage,
    needsFullscreen,
    handleBeforeUnload,
    handleVisibilityChange,
    handleFullscreenChange,
  ]);

  /* --------------------------------------------------------------------------
      Auto-capture reference image when ready (only on idle)
      -------------------------------------------------------------------------- */
  useEffect(() => {
    // Only run if the environment is ready AND we haven't already captured a valid image
    if (
        stage === "idle" &&
        resumeParsed &&
        token &&
        imageStatus === "pending" && // Only run if status is pending
        previewVideoRef.current &&
        previewCanvasRef.current
    ) {
      // Attempt auto-capture to ensure the image is ready for the user click
      captureReferenceImage().catch((e) => {
        console.warn("Auto-capture failed:", e?.message || e);
        // captureReferenceImage already sets the error state and imageStatus to 'error'
      });
    }
  }, [stage, resumeParsed, token, imageStatus, captureReferenceImage]);

  /* -------------------------
      Render
      ------------------------- */

  // Helper to determine the status message and style
 const getImageStatusIndicator = () => {
  switch (imageStatus) {
    case "captured":
      return {
        text: <><CheckCircle size={14} className="inline mr-1" /> Ready</>,
        className: "bg-emerald-50 text-emerald-800 border-emerald-300",
      };
    case "capturing":
      return {
        text: <><Loader2 size={14} className="inline mr-1 animate-spin" /> Capturing...</>,
        className: "bg-indigo-50 text-indigo-800 border-indigo-300",
      };
    case "error":
      return {
        text: <><AlertCircle size={14} className="inline mr-1" /> Failed - Click to Retry</>,
        className: "bg-rose-50 text-rose-800 border-rose-300 cursor-pointer",
      };
    case "pending":
    default:
      return {
        text: 'Initializing Camera...',
        className: "bg-slate-100 text-slate-600 border-slate-200",
      };
  }
};
  const statusIndicator = getImageStatusIndicator();

  return (
    <div className="min-h-screen bg-slate-50 py-12">
      {/* Fixed Camera View (during interview) (unchanged) */}
      {cameraActive && stage === "running" && (
        <div className="fixed top-4 right-4 z-40 w-40 h-30 bg-white rounded-xl shadow-xl border-4 border-white overflow-hidden transform scale-x-[-1] transition-transform duration-300">
          <video
            ref={proctorVideoRef}
            autoPlay
            muted
            playsInline
            className="w-full h-full object-cover"
          />
          {/* hidden canvases */}
          <canvas ref={captureCanvasRef} style={{ display: "none" }} />
          <div className="absolute top-2 left-2 flex items-center gap-1 bg-black/50 px-2 py-0.5 rounded-full">
            <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
            <span className="text-[10px] text-white font-bold tracking-wider">
              REC
            </span>
          </div>
        </div>
      )}

      <div className="max-w-6xl mx-auto">
        {/* Violation banners (unchanged) */}
        {showViolationWarning && !terminatedByViolation && (
          <div className="mb-4 p-4 rounded-xl bg-amber-50 border-2 border-amber-300 text-amber-900 flex items-start gap-3 shadow animate-in fade-in slide-in-from-top-2">
            <AlertCircle size={20} className="shrink-0" />
            <div>
              <div className="font-bold">Warning — Do not change screen</div>
              <div className="text-sm">
                We detected that you switched away from the interview or exited
                fullscreen:{" "}
                <span className="font-medium">{violationReason}</span>. This is a
                formal warning. You must re-enter fullscreen within 30 seconds or
                the interview will be terminated.
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
                The interview was terminated because you changed the screen or
                exited fullscreen after a previous warning:
                <span className="font-medium"> {violationReason}</span>. Your
                session has ended. Contact the administrator if you think this
                was an error.
              </div>
            </div>
          </div>
        )}

        {/* Fullscreen prompt modal (initial start) (unchanged) */}
        {fullscreenPromptVisible && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
            <div className="max-w-lg w-full bg-white rounded-xl p-6 shadow-lg">
              <h3 className="text-xl font-bold mb-2">
                Enter Full Screen to Begin
              </h3>
              <p className="mb-4 text-sm text-slate-700 leading-relaxed">
                For exam integrity we require the interview to run in fullscreen.
                When you enter fullscreen we will lock the interview flow to this
                window. Please click{" "}
                <strong>Enter Fullscreen & Start</strong>. If your browser blocks
                fullscreen, follow its instructions or press <kbd>F11</kbd>. If
                you prefer not to use fullscreen, you may choose{" "}
                <strong>Start anyway (not recommended)</strong> but this may
                limit your eligibility.
              </p>

              <div className="flex justify-between items-center gap-3">
                <div className="text-sm text-slate-600">
                  Fullscreen is strongly recommended to protect test integrity.
                </div>

                <div className="flex items-center gap-3">
                  <button
                    className="px-4 py-2 rounded border"
                    onClick={async () => {
                      setFullscreenPromptVisible(false);
                      setNeedsFullscreen(false);
                      await handleStart("Technical Interview", "medium", "");
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
                        await handleStart("Technical Interview", "medium", "");
                      } else {
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

        {/* Re-enter fullscreen modal after first warning (strict) (unchanged) */}
        {reenterPromptVisible && !terminatedByViolation && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
            <div className="max-w-xl w-full bg-white rounded-xl p-6 shadow-lg">
              <h3 className="text-xl font-bold mb-2 text-rose-700">
                Immediate Action Required — Re-enter Full Screen
              </h3>

              <p className="mb-3 text-sm text-slate-700 leading-relaxed">
                We detected activity that may indicate you left the interview
                window: <strong>{violationReason}</strong>. For the integrity of
                this assessment you must re-enter fullscreen within the countdown
                below. If you do not re-enter fullscreen within the allotted time
                the interview will be terminated and flagged. Please follow the
                steps below to re-enter fullscreen, or choose to end the
                interview now.
              </p>

              <div className="mb-4 flex items-center justify-between gap-4">
                <div className="flex items-center gap-3">
                  <div className="text-3xl font-bold text-rose-600">
                    {countdown}s
                  </div>
                  <div className="text-sm text-slate-600">
                    remaining to re-enter fullscreen
                  </div>
                </div>

                <div className="flex items-center gap-3">
                  <button
                    className="px-4 py-2 rounded border"
                    onClick={async () => {
                      try {
                        if (countdownTimerRef.current) {
                          window.clearInterval(countdownTimerRef.current);
                          countdownTimerRef.current = null;
                        }
                        endingRef.current = true;
                        setTerminatedByViolation(true);
                        stopCamera(); // Stop camera
                        await endInterview?.(
                          "Candidate chose to end interview after warning",
                          true
                        );
                      } catch (e) {
                        console.warn(
                          "endInterview error from reenter modal:",
                          e
                        );
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
                      const entered = await tryRequestFullscreen();
                      if (entered) {
                        if (countdownTimerRef.current) {
                          window.clearInterval(countdownTimerRef.current);
                          countdownTimerRef.current = null;
                        }
                        setReenterPromptVisible(false);
                        setShowViolationWarning(false);
                        setCountdown(30);
                      } else {
                        setViolationReason(
                          "Fullscreen blocked or not supported — try pressing F11 or allowing fullscreen in your browser."
                        );
                      }
                    }}
                  >
                    Re-enter Fullscreen Now
                  </button>
                </div>
              </div>

              <div className="text-xs text-slate-500">
                If the button does not work, try pressing <kbd>F11</kbd>{" "}
                (Windows/Linux) or <kbd>Ctrl+Command+F</kbd> (Mac) or allow
                fullscreen from your browser prompt.
              </div>
            </div>
          </div>
        )}

        {/* Header (unchanged) */}
        <div className="mb-8 flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold flex items-center gap-3 text-slate-900">
              <div className="p-2 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-xl">
                <Sparkles className="text-white" size={28} />
              </div>
              AI Technical Interview
            </h1>
            <p className="text-slate-600 mt-1 ml-14">
              Deep technical assessment powered by AI
            </p>
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

        {/* Performance Metrics Banner (during interview) (unchanged) */}
        {stage === "running" && performanceMetrics && (
          <div className="mb-6 bg-white rounded-xl shadow-md border border-slate-200 p-5 animate-in fade-in slide-in-from-top-4 duration-500">
            <div className="flex items-center justify-between flex-wrap gap-4">
              <div className="flex items-center gap-2">
                <Target size={20} className="text-indigo-600" />
                <span className="font-bold text-slate-800">
                  Performance Metrics
                </span>
              </div>

              <div className="flex items-center gap-6 flex-wrap">
                <div className="text-center">
                  <div className="text-2xl font-bold text-slate-900">
                    {performanceMetrics.question_count}
                  </div>
                  <div className="text-xs text-slate-500 uppercase tracking-wide">
                    Questions
                  </div>
                </div>

                <div className="text-center">
                  <div className="text-2xl font-bold text-indigo-600">
                    {Math.round(performanceMetrics.average_score * 100)}%
                  </div>
                  <div className="text-xs text-slate-500 uppercase tracking-wide">
                    Average
                  </div>
                </div>

                {performanceMetrics.last_score !== null && (
                  <div className="text-center">
                    <div className="text-2xl font-bold text-slate-900">
                      {Math.round(performanceMetrics.last_score * 100)}%
                    </div>
                    <div className="text-xs text-slate-500 uppercase tracking-wide">
                      Last Score
                    </div>
                  </div>
                )}

                <div className="flex items-center gap-2">
                  {renderTrendIcon(performanceMetrics.trend)}
                  <div className="text-center">
                    <div className="text-sm font-bold text-slate-900 capitalize">
                      {performanceMetrics.trend.replace("_", " ")}
                    </div>
                    <div className="text-xs text-slate-500 uppercase tracking-wide">
                      Trend
                    </div>
                  </div>
                </div>

                {(performanceMetrics.consecutive_wins > 0 ||
                  performanceMetrics.consecutive_fails > 0) && (
                  <div className="text-center">
                    <div
                      className={`text-xl font-bold ${
                        performanceMetrics.consecutive_wins > 0
                          ? "text-green-600"
                          : "text-rose-600"
                      }`}
                    >
                      {performanceMetrics.consecutive_wins > 0 ? (
                        <>🔥 {performanceMetrics.consecutive_wins}</>
                      ) : (
                        <>⚠️ {performanceMetrics.consecutive_fails}</>
                      )}
                    </div>
                    <div className="text-xs text-slate-500 uppercase tracking-wide">
                      {performanceMetrics.consecutive_wins > 0
                        ? "Win Streak"
                        : "Need Improvement"}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Not Logged In Warning (unchanged) */}
        {!token && (
          <div className="mb-6 p-4 rounded-xl bg-amber-50 border-2 border-amber-200 text-sm text-amber-900 flex items-center gap-3 shadow-sm">
            <AlertCircle size={20} className="shrink-0" />
            <span>
              You must be logged in to upload a resume or run the interview.
              <span className="ml-2 font-bold">
                <Link href="/Auth/login" className="underline hover:text-amber-700">
                  Log in
                </Link>{" "}
                or{" "}
                <Link href="/Auth/signup" className="underline hover:text-amber-700">
                  Sign up
                </Link>
                .
              </span>
            </span>
          </div>
        )}

        {/* Resume Uploader (Updated to NOT pass isFaceRegistered) */}
        {stage !== "running" && stage !== "done" && (
          <div className="mb-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
            {/* isFaceRegistered prop REMOVED here */}
            <ResumeUploader onReady={onResumeReady} onStart={handleStart} />
          </div>
        )}

        {/* Start Button & Camera Preview */}
     {stage === "idle" && resumeParsed && token && (
  <div className="mb-8 flex flex-col items-center">
    <div className="mb-6 relative group">
      <div className="w-80 h-60 bg-slate-900 rounded-2xl overflow-hidden border-4 border-white shadow-xl ring-4 ring-indigo-100 relative">
        <video
          ref={previewVideoRef}
          autoPlay
          muted
          playsInline
          className="w-full h-full object-cover transform scale-x-[-1]"
        />
        
        <canvas ref={previewCanvasRef} style={{ display: "none" }} />
        <canvas ref={captureCanvasRef} style={{ display: "none" }} />

        {cameraError && (
          <div
            className="absolute inset-0 flex flex-col items-center justify-center text-rose-100 text-sm p-6 text-center bg-black/80 cursor-pointer hover:bg-black/70 transition-colors"
            onClick={async () => {
              setCameraError(null);
              setImageStatus("pending");
              try {
                await captureReferenceImage();
              } catch (err) {
                console.warn("Manual retry failed:", err);
              }
            }}
          >
            <AlertCircle size={32} className="mb-2" />
            <div className="font-bold mb-1">{cameraError}</div>
            <div className="text-xs mt-2 underline">Click here to retry</div>
          </div>
        )}
      </div>

      {/* Status indicator */}
      <div
        className={`absolute -bottom-3 left-1/2 -translate-x-1/2 px-3 py-1 rounded-full shadow-md text-xs font-bold whitespace-nowrap border ${
          getImageStatusIndicator().className
        }`}
        onClick={imageStatus === "error" ? async () => {
          setCameraError(null);
          setImageStatus("pending");
          try {
            await captureReferenceImage();
          } catch (err) {}
        } : undefined}
      >
        {getImageStatusIndicator().text}
      </div>
    </div>

    <button
      onClick={handleStart}
      disabled={loading || imageStatus !== "captured" || startAttemptRef.current}
      className="group relative inline-flex items-center gap-3 px-10 py-5 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-2xl font-bold text-xl shadow-2xl hover:shadow-indigo-300 hover:scale-105 transition-all disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
    >
      {loading || startAttemptRef.current ? (
        <>
          <div className="w-6 h-6 border-3 border-white/30 border-t-white rounded-full animate-spin" />
          <span>Starting Interview...</span>
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

        {/* Error Message (unchanged) */}
        {error && (
          <div className="mb-6 p-4 rounded-xl bg-rose-50 border-2 border-rose-200 text-rose-700 text-sm flex items-center gap-3 shadow-sm animate-in fade-in slide-in-from-top-2 duration-300">
            <X size={20} className="shrink-0" />
            <span className="font-medium">{error}</span>
          </div>
        )}

        {/* ACTIVE INTERVIEW (unchanged) */}
        {stage === "running" && currentQuestion && !terminatedByViolation && (
          <div className="space-y-6 max-w-5xl mx-auto">
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

            <div className="bg-white rounded-2xl shadow-2xl border-2 border-slate-200 overflow-hidden">
              <div className="bg-gradient-to-r from-indigo-50 to-purple-50 p-6 border-b-2 border-slate-200">
                <div className="flex justify-between items-start mb-3">
                  <span className="text-xs font-black tracking-widest text-indigo-600 uppercase bg-indigo-100 px-3 py-1.5 rounded-lg border-2 border-indigo-200">
                    Question {history.length + 1}
                  </span>

                  <div className="flex items-center gap-2">
                    {currentQuestion.difficulty && (
                      <span
                        className={`text-xs font-bold px-2.5 py-1 rounded-lg ${
                          currentQuestion.difficulty === "expert" ||
                          currentQuestion.difficulty === "hard"
                            ? "bg-rose-100 text-rose-700 border-2 border-rose-200"
                            : "bg-amber-100 text-amber-700 border-2 border-amber-200"
                        }`}
                      >
                        {currentQuestion.difficulty.toUpperCase()}
                      </span>
                    )}
                  </div>
                </div>

                <h2 className="text-2xl font-bold text-slate-900 leading-snug">
                  {currentQuestion.questionText}
                </h2>

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

              <div className="bg-slate-50 p-8 border-t border-slate-200">
                <form onSubmit={handleSubmitAnswer}>
                  {currentQuestion.expectedAnswerType === "code" ||
                  currentQuestion.expectedAnswerType === "architectural" ? (
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

        {/* FINAL RESULTS (unchanged) */}
        {stage === "done" && (
          <div className="max-w-4xl mx-auto animate-in fade-in zoom-in duration-500">
            <div className="p-10 rounded-3xl bg-white border-2 border-slate-200 shadow-2xl">
              <div className="text-center mb-10">
                <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-gradient-to-br from-green-400 to-emerald-600 text-white mb-5 shadow-lg">
                  <CheckCircle size={40} />
                </div>
                <h2 className="text-4xl font-black text-slate-900 mb-2">
                  Interview Complete
                </h2>
                <p className="text-slate-600 text-lg">
                  Here's your comprehensive performance analysis
                </p>
              </div>

              {performanceMetrics && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                  <div className="text-center p-4 bg-slate-50 rounded-xl border border-slate-200">
                    <div className="text-3xl font-black text-slate-900">
                      {performanceMetrics.question_count}
                    </div>
                    <div className="text-xs text-slate-500 uppercase tracking-wide mt-1">
                      Questions
                    </div>
                  </div>
                  <div className="text-center p-4 bg-indigo-50 rounded-xl border border-indigo-200">
                    <div className="text-3xl font-black text-indigo-600">
                      {Math.round(performanceMetrics.average_score * 100)}%
                    </div>
                    <div className="text-xs text-slate-500 uppercase tracking-wide mt-1">
                      Avg Score
                    </div>
                  </div>
                  <div className="text-center p-4 bg-purple-50 rounded-xl border border-purple-200">
                    <div className="text-3xl font-black text-purple-600 capitalize">
                      {performanceMetrics.trend}
                    </div>
                    <div className="text-xs text-slate-500 uppercase tracking-wide mt-1">
                      Trend
                    </div>
                  </div>
                  <div className="text-center p-4 bg-emerald-50 rounded-xl border border-emerald-200">
                    <div className="text-3xl font-black text-emerald-600">
                      {Math.round(performanceMetrics.confidence * 100)}%
                    </div>
                    <div className="text-xs text-slate-500 uppercase tracking-wide mt-1">
                      Confidence
                    </div>
                  </div>
                </div>
              )}

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
                      Decision Confidence:{" "}
                      <span className="font-bold text-slate-700">
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

            {confirmRestartVisible && (
              <div className="mt-6 max-w-2xl mx-auto p-6 bg-white rounded-xl border border-slate-200 shadow-md">
                <div className="flex justify-between items-start">
                  <div>
                    <h4 className="font-bold text-lg">
                      Start a new interview?
                    </h4>
                    <p className="text-sm text-slate-600">
                      This will clear current progress and begin a fresh session.
                      Are you sure you want to continue?
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
                        handleStart();
                      }}
                    >
                      Yes, start new
                    </button>
                  </div>
                </div>
              </div>
            )}

            {showReport && (
              <div className="mt-10 space-y-5">
                <h3 className="font-black text-2xl text-slate-900 px-2 flex items-center gap-3">
                  <div className="w-1 h-8 bg-indigo-600 rounded-full"></div>
                  Complete Transcript
                </h3>

                {history.map((h, idx) => (
                  <div
                    key={idx}
                    className="bg-white p-6 rounded-2xl border-2 border-slate-200 shadow-md hover:shadow-xl transition-shadow"
                  >
                    <div className="flex gap-5">
                      <div className="shrink-0 w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 text-white flex items-center justify-center font-black text-sm shadow-md">
                        Q{idx + 1}
                      </div>
                      <div className="flex-1">
                        <div className="font-bold text-slate-900 mb-3 text-lg">
                          {h.q.questionText}
                        </div>

                        <div className="bg-slate-50 p-4 rounded-xl text-slate-700 text-sm mb-4 border-2 border-slate-100 font-mono">
                          {String(h.a)}
                        </div>

                        {h.result && (
                          <div className="space-y-3">
                            <div className="flex items-center gap-4 flex-wrap">
                              <div className="flex items-center gap-2">
                                <span className="text-xs text-slate-500 font-medium">
                                  Overall Score:
                                </span>
                                {renderScoreBadge(
                                  h.result.score || h.result.overall_score
                                )}
                              </div>

                              {h.result.verdict && (
                                <div className="flex items-center gap-2">
                                  <span className="text-xs text-slate-500 font-medium">
                                    Verdict:
                                  </span>
                                  <span
                                    className={`text-xs font-bold px-2 py-1 rounded ${
                                      h.result.verdict === "exceptional" ||
                                      h.result.verdict === "strong"
                                        ? "bg-green-100 text-green-800"
                                        : h.result.verdict === "acceptable"
                                        ? "bg-blue-100 text-blue-800"
                                        : h.result.verdict === "weak"
                                        ? "bg-amber-100 text-amber-800"
                                        : "bg-rose-100 text-rose-800"
                                    }`}
                                  >
                                    {h.result.verdict.toUpperCase()}
                                  </span>
                                </div>
                              )}
                            </div>

                            {h.result.rationale && (
                              <div className="text-xs text-slate-600 bg-blue-50 p-3 rounded-lg border border-blue-100">
                                <span className="font-bold text-blue-900">
                                  Rationale:{" "}
                                </span>
                                {h.result.rationale}
                              </div>
                            )}

                            {h.result.red_flags_detected &&
                              h.result.red_flags_detected.length > 0 && (
                                <div className="text-xs text-rose-700 bg-rose-50 p-3 rounded-lg border border-rose-200">
                                  <span className="font-bold">
                                    ⚠️ Red Flags:{" "}
                                  </span>
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