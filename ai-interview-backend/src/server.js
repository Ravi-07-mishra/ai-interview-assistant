// server.js - Revised: unified AI client, normalized fields, small bugfixes & logging
require("dotenv").config();
const express = require("express");
const axios = require("axios");
const cors = require("cors");
const multer = require("multer");
const FormData = require("form-data");
const mongoose = require("mongoose");
const { v4: uuidv4 } = require("uuid");
const helmet = require("helmet");
const rateLimit = require("express-rate-limit");

// Import auth functions - adjust path as needed
const { signupUser, loginUser, verifyToken } = require("../Controller/auth");

const app = express();

// Import models - adjust paths as needed
const User = require("../models/User");
const Session = require("../models/Session");
const QA = require("../models/QA");
const Resume = require("../models/Resume");
const Decision = require("../models/Decision");

// ---------- CONFIG ----------
const corsOrigins = process.env.CORS_ORIGIN ?
  process.env.CORS_ORIGIN.split(",") :
  ["http://localhost:3000", "http://localhost:4000"];

const AI_URL = (process.env.AI_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
const AI_API_KEY = process.env.AI_API_KEY || null;

// If AI expects raw base64 (without data:image/... prefix), set this.
// If your AI expects full data URLs, set to false.
const AI_EXPECTS_RAW_BASE64 = false;

// ---------- MIDDLEWARE ----------
app.use(cors({
  origin: corsOrigins,
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization']
}));

app.use(helmet());
// Increased limit to 50mb to handle base64 image strings from camera
app.use(express.json({ limit: "50mb" }));
app.use(express.urlencoded({ extended: true, limit: "50mb" }));

// Request logger
app.use((req, res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.path} - ip=${req.ip}`);
  next();
});

// Rate limiter
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: parseInt(process.env.RATE_LIMIT_MAX || "300", 10),
});
app.use(limiter);

// Multer config
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    const allowedExt = [".pdf", ".docx", ".txt"];
    const allowedMime = [
      "application/pdf",
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
      "text/plain"
    ];
    const name = (file.originalname || "").toLowerCase();
    const okName = allowedExt.some(ext => name.endsWith(ext));
    const okMime = allowedMime.includes(file.mimetype);
    if (okName && okMime) cb(null, true);
    else {
      const msg = `Only PDF, DOCX or TXT files are allowed. Got '${file.originalname}' (${file.mimetype})`;
      console.warn("🚫 Upload rejected:", msg);
      cb(new Error(msg));
    }
  },
});

// ---------- AI CLIENT (axios instance) ----------
const aiClient = axios.create({
  baseURL: AI_URL,
  timeout: 30000,
  headers: AI_API_KEY ? { Authorization: `Bearer ${AI_API_KEY}` } : {}
});

// ---------- IMAGE VALIDATION / NORMALIZATION HELPERS ----------
const DATA_IMAGE_RE = /^data:image\/(png|jpeg|jpg|webp);base64,([A-Za-z0-9+/]+=*)$/i;
function isValidDataImage(str, minLen = 300) {
  if (!str || typeof str !== "string") return false;
  if (str.length < minLen) return false;
  if (!str.toLowerCase().startsWith("data:image/")) return false;
  return DATA_IMAGE_RE.test(str);
}
function stripDataPrefix(dataUrl) {
  if (!dataUrl || typeof dataUrl !== "string") return null;
  const m = dataUrl.match(DATA_IMAGE_RE);
  if (!m) return null;
  return m[2]; // base64 payload only
}

// helper wrapper with retries (keeps previous behaviour)
// NOTE: instrumented to log image payload samples before forwarding to AI
async function callWithRetry(path, payload, opts = {}, attempts = 2, backoffMs = 300) {
  let lastErr;
  for (let i = 0; i < attempts; i++) {
    try {
      console.log(`📡 Calling AI endpoint: ${path}`);
      // If calling with an image, print a short sample so debugging shows whether a bad value is forwarded
      if (payload && (payload.image || payload.current_image || payload.reference_image)) {
        try {
          const imgField = payload.image ? 'image' : (payload.current_image ? 'current_image' : 'reference_image');
          const sample = String(payload[imgField]).substring(0, 80);
          console.warn(`   → payload.${imgField} sample: ${sample}... (len=${String(payload[imgField]).length})`);
        } catch (e) { /* ignore sample logging errors */ }
      }
      const resp = await aiClient.post(path, payload, { timeout: opts.timeout || 30000 });
      return resp.data;
    } catch (err) {
      lastErr = err;
      console.warn(`⚠️ AI call failed (${path}) attempt ${i + 1}/${attempts}:`, err.message);
      if (i < attempts - 1) {
        await new Promise(r => setTimeout(r, backoffMs * (i + 1)));
      }
    }
  }
  throw lastErr;
}

async function callAiGenerateQuestion(payload) { return callWithRetry("/generate_question", payload, {}, 2, 300); }
async function callAiScoreAnswer(payload) { return callWithRetry("/score_answer", payload, {}, 2, 300); }
async function callAiProbe(payload) { return callWithRetry("/probe", payload, {}, 2, 300); }
async function callAiFinalizeDecision(payload) { return callWithRetry("/finalize_decision", payload, {}, 2, 300); }
// NEW: AI face registration call
async function callAiRegisterFace(payload) {
  // Face registration can be faster, use a shorter timeout
  return callWithRetry("/interview/register-face", payload, { timeout: 10000 }, 1, 0);
}

// ---------------- DB ----------------
async function connectDB() {
  const uri = process.env.MONGO_URI || "mongodb://127.0.0.1:27017/interviewdb";
  try {
    await mongoose.connect(uri, { autoIndex: false });
    console.log("✅ Connected to MongoDB:", uri);
  } catch (error) {
    console.error("❌ MongoDB connection failed:", error);
    throw error;
  }
}

// DB helpers (kept largely as-is)
async function createSessionDB(userId = null, metadata = {}) {
  const sessionId = uuidv4();
  const s = await Session.create({
    sessionId,
    userId: userId || null,
    metadata,
    status: "active",
    qaIds: [],
    events: [],
    startedAt: new Date()
  });
  return s.toObject ? s.toObject() : s;
}

async function getSessionByIdDB(sessionId) {
  return Session.findOne({ sessionId }).lean();
}

async function markSessionCompletedDB(sessionId, extras = {}) {
  return Session.findOneAndUpdate(
    { sessionId },
    { $set: { status: "completed", endedAt: new Date(), ...extras } },
    { new: true }
  ).lean();
}

// Ensure expectedAnswerType field name is consistent here
async function createQARecordDB(sessionId, questionText, ideal_outline = null, expectedAnswerType = "short", difficulty = "medium", userId = null, metadata = {}) {
  const qaId = uuidv4();
  const questionId = uuidv4();
  const rec = await QA.create({
    qaId,
    questionId,
    sessionId,
    userId: userId || null,
    questionText,
    ideal_outline,
    expectedAnswerType,
    difficulty,
    metadata,
    askedAt: new Date(),
  });
  await Session.updateOne({ sessionId }, { $push: { qaIds: qaId } });
  return rec.toObject ? rec.toObject() : rec;
}

async function updateQARecordDB(qaId, patch) {
  const updated = await QA.findOneAndUpdate({ qaId }, { $set: patch }, { new: true });
  return updated ? (updated.toObject ? updated.toObject() : updated) : null;
}

async function getQAByQaId(qaId) {
  return QA.findOne({ qaId }).lean();
}

async function buildQuestionHistory(sessionId) {
  const sessionDoc = await Session.findOne({ sessionId }).lean();
  const qaIds = (sessionDoc?.qaIds || []).slice(-12);
  const qaDocs = await QA.find({ qaId: { $in: qaIds } }).sort({ askedAt: 1 }).lean();
  return qaDocs.map(r => ({
    question: r.questionText,
    questionText: r.questionText,
    answer: r.candidateAnswer || "",
    score: (typeof r.score === "number") ? r.score : 0,
    verdict: r.verdict || null,
    ideal_outline: r.ideal_outline || "",
  }));
}

// ---------------- AUTH MIDDLEWARE ----------------
// verifyToken is expected to behave like express middleware (req, res, next)
function requireAuth(req, res, next) {
  return verifyToken(req, res, next);
}

// ---------- ROUTES ----------

// Health
app.get("/health", (req, res) => {
  res.json({
    status: "backend running",
    ai_service: AI_URL,
    timestamp: new Date().toISOString()
  });
});

app.get("/test", (req, res) => res.json({ message: "Server is working!" }));

// Auth
app.post("/auth/signup", async (req, res) => {
  try {
    console.log("📝 Signup request:", req.body?.email);
    const { name, email, password } = req.body;
    const user = await signupUser({ name, email, password });
    const { token } = await loginUser({ email, password });
    return res.status(201).json({
      token,
      user: { id: user._id, name: user.name, email: user.email }
    });
  } catch (err) {
    console.error("❌ Signup error:", err.message);
    return res.status(400).json({ message: err.message || "signup failed" });
  }
});

// Register face (explicit endpoint used by frontend)
// Body: { sessionId, image }
// Replace the whole /interview/register-face handler with this:
app.post("/interview/register-face", requireAuth, async (req, res) => {
  try {
    const { sessionId, image } = req.body || {};
    if (!sessionId || !image) return res.status(400).json({ error: "sessionId and image required" });

    // Quick client-side sanity check
    if (!isValidDataImage(image, 200)) {
      console.warn(`🚫 register-face rejected: invalid image (len=${String(image).length}) sample=${String(image).substring(0,36)}`);
      return res.status(400).json({ error: "invalid_image", message: "Image must be a data:image/...;base64 string" });
    }

    const session = await Session.findOne({ sessionId }).lean();
    if (!session) return res.status(404).json({ error: "session_not_found" });

    // 1) First, call AI service to validate/register the reference image.
    //    Only if the AI confirms registration do we persist the reference image locally.
    try {
      const aiPayload = {
        sessionId,
        image: AI_EXPECTS_RAW_BASE64 ? stripDataPrefix(image) : image
      };
      console.log(`📸 register-face: calling AI for session ${sessionId} (imageLen=${image.length})`);
      const aiResp = await callAiRegisterFace(aiPayload);

      // aiResp shape varies by implementation; check common success cases
      // Accept if aiResp.status === 'registered' or aiResp.ok === true
      const registered = (aiResp && (aiResp.status === "registered" || aiResp.ok === true || aiResp.result === "registered"));

      if (!registered) {
        // If AI rejected the image, return 400 and include reason if available
        const reason = aiResp?.message || aiResp?.detail || aiResp?.error || "AI rejected the reference image";
        console.warn(`⚠️ AI register-face rejected image for session ${sessionId}:`, reason);
        return res.status(400).json({ ok: false, error: "ai_rejected_image", message: String(reason) });
      }

      // 2) Only persist the reference image after AI accepted it
      await Session.updateOne({ sessionId }, { $set: { "metadata.referenceFace": image, "metadata.referenceRegisteredAt": new Date() } });

      // Response: success
      console.log(`✅ Reference face registered and persisted for session ${sessionId}`);
      return res.json({ ok: true, status: "registered", message: "Reference image registered and saved" });

    } catch (aiErr) {
      // If AI call failed (network / service error), return 502 so frontend knows it's an external error
      console.warn("AI register-face failed:", aiErr?.message || aiErr);
      return res.status(502).json({ ok: false, error: "ai_service_error", message: "Failed to register face with verification service" });
    }

  } catch (err) {
    console.error("register-face error:", err);
    return res.status(500).json({ error: "register_face_failed", details: err?.message });
  }
});

app.post("/auth/login", async (req, res) => {
  try {
    console.log("🔐 Login request:", req.body?.email);
    const { email, password } = req.body;
    const { user, token } = await loginUser({ email, password });
    return res.json({
      token,
      user: { id: user._id, name: user.name, email: user.email }
    });
  } catch (err) {
    console.error("❌ Login error:", err.message);
    return res.status(401).json({ message: err.message || "invalid credentials" });
  }
});

// Resume processing
app.post("/process-resume", requireAuth, upload.single("file"), async (req, res) => {
  try {
    console.log("📄 Processing resume:", req.file?.originalname);
    if (!req.file) return res.status(400).json({ error: "No file uploaded" });

    const form = new FormData();
    form.append("file", req.file.buffer, {
      filename: req.file.originalname,
      contentType: req.file.mimetype || "application/octet-stream",
      knownLength: req.file.size,
    });

    const aiResp = await aiClient.post("/parse_resume", form, {
      headers: { ...form.getHeaders() },
      maxContentLength: Infinity,
      maxBodyLength: Infinity,
      timeout: 30000,
    });

    const parsed = aiResp.data?.parsed ?? aiResp.data ?? null;
    console.log("✅ Resume parsed successfully");

    try {
      const resumeDoc = await Resume.create({
        userId: req.userId || null,
        sourceUrl: null,
        parsed,
        redactionLog: parsed?.redaction_log || [],
        rawTextStored: false,
        createdAt: new Date()
      });
      return res.status(201).json({ parsed, resumeId: resumeDoc._id });
    } catch (e) {
      console.warn("⚠️ Resume save failed:", e.message);
      return res.json({ parsed });
    }
  } catch (err) {
    console.error("❌ Parse Resume Error:", err.message);
    const details = err?.response?.data ?? err?.message ?? String(err);
    return res.status(500).json({
      error: "failed_to_parse_resume",
      details: process.env.NODE_ENV === "production" ? undefined : details
    });
  }
});

// Interview start
app.post("/interview/start", requireAuth, async (req, res) => {
    // FIX 1: Define 'session' outside the try block so it's accessible in the catch block
    let session = null; 
    
    try {
        console.log("🎬 Starting interview for user:", req.userId);
        const body = req.body || {};
        const userId = req.userId || null;
        const referenceImage = body.referenceImage || null; // Capture reference image from frontend body

        // --- STEP 1: VALIDATION ---
        if (!referenceImage || !isValidDataImage(referenceImage, 200)) {
            console.warn(`🚫 /interview/start: REJECTED - Missing or invalid referenceImage (len=${String(referenceImage).length})`);
            return res.status(400).json({ error: "invalid_reference_image", message: "A valid reference image is required to start the interview." });
        }

        // --- STEP 2: CREATE TEMPORARY SESSION & TRY AI REGISTRATION ---
        // Assign to 'session' defined above
        session = await createSessionDB(userId, {
            from: "frontend",
            // Store reference image locally immediately, assuming client-side check passed
            referenceFace: referenceImage
        });
        console.log("📝 Created session:", session.sessionId);

        try {
            const aiRegPayload = {
                sessionId: session.sessionId,
                image: AI_EXPECTS_RAW_BASE64 ? stripDataPrefix(referenceImage) : referenceImage
            };
            console.log(`📸 start: calling AI register-face for session ${session.sessionId} (imageLen=${referenceImage.length})`);

            // This MUST throw on 400 or other errors from the AI service!
            const aiRegResp = await callAiRegisterFace(aiRegPayload); 
            
            // Check for explicit failure status from AI response body (in case AI service returns 200 but status: failed)
            if (aiRegResp?.status === 'failed' || aiRegResp?.ok === false || aiRegResp?.result === 'rejected') {
                 const reason = aiRegResp?.message || aiRegResp?.detail || aiRegResp?.error || "AI explicitly rejected the reference image";
                 throw new Error(reason); // Treat explicit rejection as failure
             }

            console.log("📸 AI Face Registration Status: registered");
            
            // Update DB to mark registration time (optional, but good practice)
            await Session.updateOne({ sessionId: session.sessionId }, { $set: { "metadata.referenceRegisteredAt": new Date() } });

        } catch (aiErr) {
            // CRITICAL BLOCK: If AI registration fails (e.g., DeepFace enforce_detection error 400)
            console.warn("⚠️ AI registration failed. ABORTING START.", aiErr?.message || aiErr);
            
            // Return a 400 error to the client with the failure reason
            const reason = aiErr?.response?.data?.message || aiErr?.message || "AI failed to detect face or register image.";
            
            // Mark the session as terminated/invalid (using session.sessionId safely)
            await Session.updateOne({ sessionId: session.sessionId }, { $set: { status: "aborted", endedReason: "AI registration failed" } });

            return res.status(400).json({ 
                error: "face_registration_failed", 
                message: `Cannot start interview: ${reason}` 
            });
        }
        // --- STEP 3: IF WE REACH HERE, REGISTRATION WAS SUCCESSFUL. PROCEED TO QUESTION GENERATION. ---

        const aiPayload = {
            request_id: uuidv4(),
            session_id: session.sessionId,
            user_id: userId || "anonymous",
            mode: "first",
            resume_summary: body.resume_summary || (body.parsed_resume?.summary) || "",
            retrieved_chunks: body.retrieved_chunks || [],
            conversation: [],
            question_history: [],
            token_budget: 3000,
            allow_pii: !!body.allow_pii,
            options: { return_prompt: false, temperature: 0.1 }
        };

        const aiResp = await callAiGenerateQuestion(aiPayload);
        const parsed = aiResp.parsed || {};
        // ... (rest of the question generation logic remains the same)

        const questionText = parsed.question || parsed.questionText ||
            "Tell me about the most technically challenging project on your resume. What specific problem did you solve, and how did you approach it?";

        const qaMetadata = {
            target_project: parsed.target_project,
            technology_focus: parsed.technology_focus,
            red_flags: parsed.red_flags || [],
            confidence: parsed.confidence
        };

        const qaDoc = await createQARecordDB(
            session.sessionId,
            questionText,
            parsed.ideal_answer_outline || parsed.ideal_outline || "",
            parsed.expected_answer_type || parsed.expectedAnswerType || "medium",
            parsed.difficulty || "hard",
            userId,
            qaMetadata
        );

        console.log("✅ Interview started with question ID:", qaDoc.questionId);

        return res.json({
            sessionId: session.sessionId,
            firstQuestion: {
                qaId: qaDoc.qaId,
                questionId: qaDoc.questionId,
                questionText: qaDoc.questionText,
                target_project: parsed.target_project,
                technology_focus: parsed.technology_focus,
                expectedAnswerType: qaDoc.expectedAnswerType || parsed.expectedAnswerType || "medium",
                difficulty: qaDoc.difficulty,
                ideal_outline: qaDoc.ideal_outline || parsed.ideal_outline,
                red_flags: parsed.red_flags
            },
            proctoring: {
                referenceRegistered: true, // Now guaranteed to be true if we reached here
                aiRegistrationStatus: "registered"
            }
        });
    } catch (err) {
        // FIX 2: Ensure logging uses session.sessionId only if session exists
        const logId = session ? session.sessionId : 'N/A (Pre-session)';
        console.error(`❌ Interview start error for session ${logId}:`, err.message);

        // Generic server error handling (500)
        const details = err?.response?.data ?? err?.message ?? String(err);
        return res.status(500).json({
            error: "failed_to_start_interview",
            details: process.env.NODE_ENV === "production" ? undefined : details
        });
    }
});

// Interview answer
app.post("/interview/answer", requireAuth, async (req, res) => {
  try {
    console.log("💬 Processing answer for session:", req.body.sessionId);

    const { sessionId, qaId, questionId, questionText } = req.body || {};
    const candidateAnswer = req.body.candidateAnswer || req.body.candidate_answer || "";
    const userId = req.userId || null;
    if (!sessionId) {
      console.error("❌ Missing sessionId");
      return res.status(400).json({ error: "missing sessionId" });
    }
    if (!qaId && !questionId) {
      console.error("❌ Missing qaId/questionId");
      return res.status(400).json({ error: "missing qaId or questionId" });
    }

    // find QA
    let qaRec = null;
    if (qaId) qaRec = await getQAByQaId(qaId);
    else qaRec = await QA.findOne({ questionId, sessionId }).lean();

    if (!qaRec) {
      console.error("❌ QA record not found:", qaId || questionId);
      return res.status(404).json({ error: "qa_record_not_found" });
    }

    console.log("📝 Found QA record:", qaRec.qaId);

    // Save answer text (best-effort; field names preserved)
    await updateQARecordDB(qaRec.qaId, {
      candidateAnswer,
      answeredAt: new Date()
    });

    // Build history
    const questionHistory = await buildQuestionHistory(sessionId);
    console.log("📊 Question history length:", questionHistory.length);

    // Build score payload
    const scorePayload = {
      request_id: uuidv4(),
      session_id: sessionId,
      user_id: userId || "anonymous",
      question_text: questionText || qaRec.questionText,
      ideal_outline: qaRec.ideal_outline || "",
      candidate_answer: candidateAnswer,
      resume_summary: req.body.resume_summary || "",
      retrieved_chunks: req.body.retrieved_chunks || [],
      question_history: questionHistory,
      token_budget: 1200,
      allow_pii: !!req.body.allow_pii,
      options: { temperature: 0.0 }
    };

    const aiScoreResp = await callAiScoreAnswer(scorePayload);
    // normalize AI score fields (support both snake_case and camelCase)
    const validated = aiScoreResp.validated || aiScoreResp.validation || {};
    const overallScore = validated.overall_score ?? validated.score ?? (validated.overallScore ?? 0);
    const rubricScores = validated.dimension_scores || validated.rubric_scores || validated.dimensionScores || null;
    const verdict = validated.verdict || "weak";
    const confidence = validated.confidence ?? 0.5;

    console.log("📊 Score received:", overallScore);

    // Update QA with score and normalized fields
    const scoreUpdate = {
      gradedBy: "llm",
      score: overallScore,
      rubricScores,
      verdict,
      confidence,
      rationale: validated.rationale || aiScoreResp.rationale || "",
      improvement: validated.mentor_tip || validated.follow_up_probe || null, // Check for mentor_tip from AI service
      red_flags_detected: validated.red_flags_detected || validated.redFlagsDetected || [],
      missing_elements: validated.missing_elements || validated.missingElements || [],
      needsHumanReview: aiScoreResp.needs_human_review || aiScoreResp.needsHumanReview || aiScoreResp.in_gray_zone || aiScoreResp.inGrayZone || false,
      gradedAt: new Date(),
      metadata: {
        ai_parse_ok: !!aiScoreResp.parse_ok,
        in_gray_zone: aiScoreResp.in_gray_zone || aiScoreResp.inGrayZone || false
      }
    };
    await updateQARecordDB(qaRec.qaId, scoreUpdate);

    // Update history array for decision making
    questionHistory.push({
      question: qaRec.questionText,
      questionText: qaRec.questionText,
      answer: candidateAnswer,
      score: scoreUpdate.score,
      verdict: scoreUpdate.verdict,
      ideal_outline: qaRec.ideal_outline || ""
    });

    // Decision & next question
    let nextQuestion = null;
    let ended = false;
    let performanceMetrics = null;

    try {
      const decisionPayload = {
        request_id: uuidv4(),
        session_id: sessionId,
        user_id: userId || "anonymous",
        resume_summary: req.body.resume_summary || "",
        conversation: req.body.conversation || [],
        question_history: questionHistory,
        retrieved_chunks: req.body.retrieved_chunks || [],
        token_budget: 800,
        allow_pii: !!req.body.allow_pii,
        accept_model_final: true
      };

      const finalizeResp = await callAiFinalizeDecision(decisionPayload);
      const decisionResult = finalizeResp.result || finalizeResp;
      performanceMetrics = finalizeResp.performance_metrics || finalizeResp.performanceMetrics || null;

      const isFinal = finalizeResp.is_final || finalizeResp.isFinal || false;
      const modelDecision = decisionResult?.parsed || decisionResult?.decision || decisionResult;

      if (isFinal && modelDecision && modelDecision.ended) {
        ended = true;
        console.log("🏁 Interview ended by model decision:", modelDecision.verdict);

        const decisionDoc = await Decision.create({
          decisionId: uuidv4(),
          sessionId,
          decidedBy: "model",
          verdict: modelDecision.verdict,
          confidence: modelDecision.confidence || 0.5,
          reason: modelDecision.reason || "",
          recommended_role: modelDecision.recommended_role || modelDecision.recommendedRole || null,
          key_strengths: modelDecision.key_strengths || modelDecision.keyStrengths || [],
          critical_weaknesses: modelDecision.critical_weaknesses || modelDecision.criticalWeaknesses || [],
          rawModelOutput: decisionResult,
          performanceMetrics,
          decidedAt: new Date()
        });

        await markSessionCompletedDB(sessionId, {
          finalDecisionRef: decisionDoc._id,
          performanceMetrics
        });
      }
    } catch (e) {
      console.warn("⚠️ Decision check failed:", e?.message || e);
    }

    // If not ended, generate probe or followup
    if (!ended) {
      try {
        const inGrayZone = aiScoreResp.in_gray_zone || aiScoreResp.inGrayZone || false;
        const shouldProbe = inGrayZone || (scoreUpdate.score < 0.60 && scoreUpdate.score >= 0.30);

        if (shouldProbe && (validated.follow_up_probe || scoreUpdate.improvement)) {
          console.log("🔍 Generating probe question");
          const probePayload = {
            request_id: uuidv4(),
            session_id: sessionId,
            user_id: userId || "anonymous",
            weakness_topic: validated.missing_elements?.[0] || "the previous topic",
            prev_question: qaRec.questionText,
            prev_answer: candidateAnswer,
            resume_summary: req.body.resume_summary || "",
            retrieved_chunks: req.body.retrieved_chunks || [],
            conversation: req.body.conversation || [],
            token_budget: 600,
            allow_pii: !!req.body.allow_pii
          };

          const probeResp = await callAiProbe(probePayload);
          const parsed = probeResp.parsed || {};
          const probeQuestion = parsed.probe_question || validated.follow_up_probe || scoreUpdate.improvement;

          const newQa = await createQARecordDB(
            sessionId,
            probeQuestion,
            null,
            parsed.expected_answer_length || parsed.expectedAnswerLength || "medium",
            parsed.difficulty || "medium",
            userId,
            { is_probe: true }
          );

          nextQuestion = {
            qaId: newQa.qaId,
            questionId: newQa.questionId,
            questionText: newQa.questionText,
            expectedAnswerType: newQa.expectedAnswerType,
            difficulty: newQa.difficulty
          };
        } else {
          console.log("➡️ Generating follow-up question");
          const genPayload = {
            request_id: uuidv4(),
            session_id: sessionId,
            user_id: userId || "anonymous",
            mode: "followup",
            resume_summary: req.body.resume_summary || "",
            retrieved_chunks: req.body.retrieved_chunks || [],
            conversation: req.body.conversation || [],
            question_history: questionHistory,
            token_budget: 1500,
            allow_pii: !!req.body.allow_pii,
            options: { temperature: 0.1 }
          };

          const genResp = await callAiGenerateQuestion(genPayload);
          const parsed = genResp.parsed || {};
          const qText = parsed.question || parsed.followup_question ||
            "Can you elaborate on your approach with a specific example?";

          const newQa = await createQARecordDB(
            sessionId,
            qText,
            parsed.ideal_answer_outline || parsed.ideal_outline || "",
            parsed.expected_answer_type || parsed.expectedAnswerType || "medium",
            parsed.difficulty || "hard",
            userId,
            {
              target_project: parsed.target_project,
              technology_focus: parsed.technology_focus,
              red_flags: parsed.red_flags || []
            }
          );

          nextQuestion = {
            qaId: newQa.qaId,
            questionId: newQa.questionId,
            questionText: newQa.questionText,
            target_project: parsed.target_project,
            technology_focus: parsed.technology_focus,
            expectedAnswerType: newQa.expectedAnswerType,
            difficulty: newQa.difficulty,
            ideal_outline: newQa.ideal_outline
          };
        }
      } catch (e) {
        console.warn("⚠️ Next question generation failed:", e?.message || e);
        ended = true;
      }
    }

    // Return latest QA
    const latestQa = await getQAByQaId(qaRec.qaId);

    const result = {
      overall_score: latestQa?.score ?? 0,
      score: latestQa?.score ?? 0,
      dimension_scores: latestQa?.rubricScores ?? null,
      rubricScores: latestQa?.rubricScores ?? null,
      verdict: latestQa?.verdict ?? "weak",
      confidence: latestQa?.confidence ?? 0.5,
      rationale: latestQa?.rationale ?? "",
      red_flags_detected: latestQa?.red_flags_detected ?? [],
      missing_elements: latestQa?.missing_elements ?? [],
      improvement: latestQa?.improvement ?? null,
      follow_up_probe: latestQa?.improvement ?? null,
    };

    console.log("✅ Answer processed successfully");

    return res.json({
      validated: result,
      result,
      nextQuestion,
      ended,
      is_final: ended,
      performance_metrics: performanceMetrics,
      needs_human_review: latestQa?.needsHumanReview || false,
      in_gray_zone: latestQa?.metadata?.in_gray_zone || false
    });

  } catch (err) {
    console.error("❌ Interview answer error:", err?.message || err);
    console.error(err?.stack || "");
    const details = err?.response?.data ?? err?.message ?? String(err);
    return res.status(500).json({
      error: "failed_to_score_answer",
      details: process.env.NODE_ENV === "production" ? undefined : details
    });
  }
});

// Record a violation
// Record a violation (safer termination decision)
app.post("/interview/violation", requireAuth, async (req, res) => {
  try {
    const { sessionId, reason, timestamp, action } = req.body || {};
    if (!sessionId) return res.status(400).json({ error: "sessionId required" });

    const ev = {
      id: uuidv4(),
      type: "violation",
      reason: reason || "screen-change",
      at: timestamp ? new Date(timestamp) : new Date(),
      by: req.userId || null,
      action: action || "warning"
    };

    // Increment and push event
    const updated = await Session.findOneAndUpdate(
      { sessionId },
      {
        $inc: { violationCount: 1 },
        $push: { events: ev }
      },
      { new: true, upsert: false }
    ).lean();

    if (!updated) return res.status(404).json({ error: "session_not_found" });

    // Robust read of violationCount
    const currentCount = (typeof updated.violationCount === "number") ? updated.violationCount : 0;
    console.log(`⚠️ Violation recorded for session ${sessionId}: ${ev.reason} (count=${currentCount}, action=${ev.action})`);

    // Decide whether to terminate. Use a threshold (2) OR explicit terminate flag after sanity-check.
    const THRESHOLD = 2;
    const explicitlyTerminate = action === "terminate";
    const shouldTerminate = (currentCount >= THRESHOLD) || explicitlyTerminate;

    if (shouldTerminate && !String(updated.status || "").startsWith("completed")) {
      try {
        const extras = {
          terminatedByViolation: true,
          endedReason: `Interview integrity failure: ${reason}`,
          status: "completed",
          endedAt: new Date()
        };

        const completedSession = await Session.findOneAndUpdate(
          { sessionId },
          { $set: extras },
          { new: true }
        ).lean();

        const decisionDoc = await Decision.create({
          decisionId: uuidv4(),
          sessionId,
          decidedBy: "system",
          verdict: "reject",
          confidence: 1.0,
          reason: extras.endedReason,
          recommended_role: null,
          key_strengths: [],
          critical_weaknesses: [],
          rawModelOutput: { terminated_by_violation: true, reason: extras.endedReason, violationCount: currentCount },
          performanceMetrics: { averageScore: null },
          decidedAt: new Date()
        });

        await Session.updateOne({ sessionId }, { $set: { finalDecisionRef: decisionDoc._id } });

        return res.json({
          ok: true,
          event: ev,
          violationCount: currentCount,
          terminated: true,
          message: `Session terminated due to repeated violations (${currentCount}).`,
          finalDecisionRef: decisionDoc._id
        });
      } catch (innerErr) {
        console.warn("⚠️ Failed to complete session after threshold:", innerErr);
        return res.json({
          ok: true,
          event: ev,
          violationCount: currentCount,
          terminated: true,
          message: `Violation threshold reached (error creating decision)`
        });
      }
    }

    return res.json({
      ok: true,
      event: ev,
      violationCount: currentCount,
      terminated: false,
      message: currentCount === 1 ? "Warning recorded" : "Violation recorded"
    });

  } catch (err) {
    console.error("❌ Violation recording failed:", err?.message || err);
    return res.status(500).json({ error: "failed_to_record_violation", details: err?.message });
  }
});

// Interview end
app.post("/interview/end", requireAuth, async (req, res) => {
  try {
    console.log("🏁 Ending interview:", req.body.sessionId);
    const { sessionId, reason, terminated_by_violation } = req.body || {};
    if (!sessionId) return res.status(400).json({ error: "missing sessionId" });

    const s = await getSessionByIdDB(sessionId);
    if (!s) return res.status(404).json({ error: "session_not_found" });

    const recs = await QA.find({ sessionId }).lean();
    const scores = recs.map(r => r.score).filter(v => v !== null && v !== undefined && !isNaN(v));
    const avgScore = scores.length ? (scores.reduce((a, b) => a + b, 0) / scores.length) : null;

    const extras = {};
    if (reason) extras.endedReason = reason;
    if (terminated_by_violation) {
      extras.terminatedByViolation = true;
      extras.finalVerdict = "reject";
      extras.finalReason = reason || "screen-change violation";
    }

    const updatedSession = await markSessionCompletedDB(sessionId, extras);

    let decisionDoc = null;
    if (terminated_by_violation) {
      try {
        decisionDoc = await Decision.create({
          decisionId: uuidv4(),
          sessionId,
          decidedBy: "system",
          verdict: "reject",
          confidence: 1.0,
          reason: extras.finalReason,
          recommended_role: null,
          key_strengths: [],
          critical_weaknesses: [],
          rawModelOutput: { terminated_by_violation: true, reason: extras.finalReason },
          performanceMetrics: { averageScore: avgScore },
          decidedAt: new Date()
        });

        await Session.findOneAndUpdate({ sessionId }, { $set: { finalDecisionRef: decisionDoc._id } });
      } catch (e) {
        console.warn("⚠️ Failed to create Decision for violation end:", e.message);
      }
    }

    console.log("✅ Interview ended successfully", { sessionId, terminated_by_violation: !!terminated_by_violation });

    return res.json({
      ok: true,
      sessionId,
      finalScore: avgScore !== null ? Math.round(avgScore * 1000) / 10 : null,
      totalQuestions: recs.length,
      terminated_by_violation: !!terminated_by_violation,
      finalDecisionRef: decisionDoc ? decisionDoc._id : null,
      session: updatedSession
    });
  } catch (err) {
    console.error("❌ Interview end error:", err?.message || err);
    return res.status(500).json({
      error: "failed_to_end_session",
      details: process.env.NODE_ENV === "production" ? undefined : err?.message
    });
  }
});

// Admin route
app.get("/admin/session/:id", requireAuth, async (req, res) => {
  try {
    const s = await Session.findOne({ sessionId: req.params.id }).lean();
    if (!s) return res.status(404).json({ error: "not_found" });
    const qas = await QA.find({ qaId: { $in: s.qaIds || [] } }).lean();
    return res.json({ session: s, qas });
  } catch (e) {
    console.error("❌ Admin/session error:", e);
    return res.status(500).json({ error: "internal_server_error" });
  }
});

// Proctoring endpoint (face verification)
app.post("/interview/proctor", requireAuth, async(req, res) => {
    try {
        const { sessionId, image } = req.body;
        if (!sessionId || !image) return res.status(400).json({ error: "Missing data" });

        const session = await Session.findOne({ sessionId }).lean();
        if (!session) return res.status(404).json({ error: "Session not found" });

        if (!isValidDataImage(image, 300)) {
            console.warn(`🚫 Proctoring rejected: invalid image for session ${sessionId} (len=${String(image).length}) sample=${String(image).substring(0,36)}`);
            return res.json({ status: "warning", message: "Live frame invalid or not a base64 data URL" });
        }

        // --- Registration Fallback Logic (validated) ---
if (!session.metadata?.referenceFace) {
  console.log(`📸 Attempting late registration for Session: ${sessionId}`);

  // sanity-check incoming live frame
  if (!isValidDataImage(image, 300)) {
    console.warn(`🚫 Proctor fallback rejected invalid image for session ${sessionId} (len=${String(image).length})`);
    return res.json({ status: "warning", message: "Live frame invalid or not a base64 data URL" });
  }

  try {
    const aiPayload = { sessionId, image: AI_EXPECTS_RAW_BASE64 ? stripDataPrefix(image) : image };
    const aiResp = await callAiRegisterFace(aiPayload);

    const registered = (aiResp && (aiResp.status === "registered" || aiResp.ok === true || aiResp.result === "registered"));

    if (!registered) {
      const reason = aiResp?.message || aiResp?.error || "AI rejected the late registration image";
      console.warn(`⚠️ AI rejected late registration for session ${sessionId}:`, reason);
      return res.json({ status: "warning", message: "Reference image rejected by verification service", detail: reason });
    }

    // Persist only after AI acceptance
    await Session.updateOne({ sessionId }, { $set: { "metadata.referenceFace": image, "metadata.referenceRegisteredAt": new Date() } });

    console.log(`✅ Late reference face successfully registered for session ${sessionId}`);
    return res.json({ status: "registered", message: "Face registered successfully" });
  } catch (e) {
    console.warn("⚠️ Failed to register face with AI service late:", e?.message || e);
    return res.json({ status: "warning", message: "Face registration failed due to AI service error" });
  }
}

        // --- End Registration Fallback ---

        const referenceImage = session.metadata.referenceFace;

        try {
            const verifyPayload = {
                reference_image: AI_EXPECTS_RAW_BASE64 ? stripDataPrefix(referenceImage) : referenceImage,
                current_image: AI_EXPECTS_RAW_BASE64 ? stripDataPrefix(image) : image
            };

            const aiResponse = await aiClient.post("/verify_face", verifyPayload, { timeout: 10000 });

            const { verified, distance, error, violation_type } = aiResponse.data || {};

            // --- CRITICAL VIOLATION CHECK: Unified Logic for No Face / Mismatch ---
            if (violation_type === "no_face_detected" || violation_type === "face_mismatch") {
                const isMismatch = (violation_type === "face_mismatch");
                const reason = isMismatch 
                    ? `Unauthorized face detected (Distance: ${typeof distance === 'number' ? distance.toFixed(4) : 'N/A'})` 
                    : "Candidate left camera view (No face detected)";
                const type = isMismatch ? "unauthorized_face" : "no_face";

                console.log(`⚠️ VIOLATION: ${reason} for session ${sessionId}`);
                await Session.updateOne(
                    { sessionId },
                    {
                        $inc: { violationCount: 1 },
                        $push: { events: {
                            id: uuidv4(),
                            type: type,
                            reason: reason,
                            at: new Date()
                        }}
                    }
                );
                // Status 'failed' triggers the frontend warning/timer/violation count display.
                return res.json({ status: "failed", verified: false, reason: reason });
            }
            // ------------------------------------

            if (error) {
                console.warn("⚠️ Face verification returned error:", error);
                return res.json({ status: "warning", message: "Face detection issue (image unclear or processing error)" });
            }
            
            // If the AI Service returns verified: false without a violation_type (Shouldn't happen after fixes)
            if (!verified) {
                // We keep this as a final fallback, though the logic above should capture it.
                console.warn(`⚠️ Unexpected Face Mismatch! Default fallback triggered. Distance: ${distance}`);
                return res.json({ status: "failed", verified: false, distance });
            }

            return res.json({ status: "success", verified: true, distance });
        }catch (aiErr) {
            console.error("❌ AI Verification Service Error:", aiErr?.message);

            // 1. If AI sent a response (e.g. 400 Bad Request), forward it exactly
            if (aiErr.response) {
                return res.status(aiErr.response.status).json(aiErr.response.data);
            }

            // 2. If AI is down/network error, return 502
            return res.status(502).json({ 
                status: "failed", 
                verified: false, 
                error: "AI service unavailable or timeout" 
            });
        }
    } catch (err) {
        console.error("❌ Proctoring Error:", err?.message || err);
        return res.status(500).json({ error: "Proctoring failed" });
    }
});


// 404 handler
app.use((req, res) => {
  console.warn("⚠️ 404 Not Found:", req.method, req.path);
  res.status(404).json({
    error: "not_found",
    message: `Route ${req.method} ${req.path} not found`,
    availableRoutes: [
      "GET /health",
      "GET /test",
      "POST /auth/signup",
      "POST /auth/login",
      "POST /process-resume",
      "POST /interview/start",
      "POST /interview/register-face",
      "POST /interview/proctor",
      "POST /interview/answer",
      "POST /interview/violation",
      "POST /interview/end"
    ]
  });
});

// Global error handler
app.use((err, req, res, next) => {
  console.error("❌ Global error:", err?.message || err);
  if (err.code === "LIMIT_FILE_SIZE") {
    return res.status(400).json({ error: "File too large" });
  }
  if (/Only PDF|DOCX|TXT/.test(err.message || "")) {
    return res.status(400).json({ error: err.message });
  }
  const details = process.env.NODE_ENV === "production" ? undefined : (err?.stack || err?.message);
  res.status(500).json({ error: "internal_server_error", details });
});

// Start server
let server;
(async function init() {
  try {
    await connectDB();
    const PORT = process.env.PORT || 4000;
    server = app.listen(PORT, '0.0.0.0', () => {
      console.log("\n" + "=".repeat(50));
      console.log("🚀 Backend Server Started");
      console.log("=".repeat(50));
      console.log(`📍 Port: ${PORT}`);
      console.log(`🤖 AI Service: ${AI_URL}`);
      console.log(`🔒 CORS Origins: ${corsOrigins.join(", ")}`);
      console.log(`📚 MongoDB: Connected`);
      console.log("=".repeat(50) + "\n");
    });
  } catch (e) {
    console.error("❌ Failed to init server:", e);
    process.exit(1);
  }
})();

process.on('SIGINT', async () => {
  console.log('\n⚠️ SIGINT received, shutting down gracefully...');
  if (server) await new Promise(r => server.close(r));
  await mongoose.disconnect();
  console.log('✅ Shutdown complete');
  process.exit(0);
});