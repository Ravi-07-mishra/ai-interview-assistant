// server.js - Fixed version with proper routing
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

// ---------------- MIDDLEWARE (ORDER MATTERS!) ----------------
// CORS must come before other middleware
const corsOrigins = process.env.CORS_ORIGIN ? 
  process.env.CORS_ORIGIN.split(",") : 
  ["http://localhost:3000", "http://localhost:4000"];

app.use(cors({
  origin: corsOrigins,
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization']
}));

app.use(helmet());
app.use(express.json({ limit: "200kb" }));
app.use(express.urlencoded({ extended: true }));

// Request logger (before routes)
app.use((req, res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.path}`);
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
    const name = (file.originalname || "").toLowerCase();
    const allowedMime = [
      "application/pdf", 
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document", 
      "text/plain"
    ];
    const okName = allowedExt.some(ext => name.endsWith(ext));
    const okMime = allowedMime.includes(file.mimetype);
    if (okName && okMime) cb(null, true);
    else cb(new Error("Only PDF, DOCX or TXT files are allowed"));
  },
});

const AI_URL = (process.env.AI_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

// ---------------- DB CONNECTION ----------------
async function connectDB() {
  const uri = process.env.MONGO_URI || "mongodb://127.0.0.1:27017/interviewdb";
  try {
    await mongoose.connect(uri);
    console.log("✅ Connected to MongoDB:", uri);
  } catch (error) {
    console.error("❌ MongoDB connection failed:", error);
    throw error;
  }
}

// ---------------- DB HELPERS ----------------
async function createSessionDB(userId = null, metadata = {}) {
  const sessionId = uuidv4();
  const s = await Session.create({ 
    sessionId, 
    userId: userId || null, 
    metadata, 
    status: "active", 
    qaIds: [],
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

async function createQARecordDB(sessionId, questionText, ideal_outline = null, expectedAnswerType = "short", difficulty = "medium", userId = null, metadata = {}) {
  const qaId = uuidv4();
  const questionId = uuidv4();
  const rec = await QA.create({
    qaId,
    sessionId,
    userId: userId || null,
    questionText,
    questionId,
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
function requireAuth(req, res, next) {
  return verifyToken(req, res, next);
}

// ---------------- AI CALL HELPERS ----------------
async function callWithRetry(url, payload, opts = {}, attempts = 2, backoffMs = 300) {
  let lastErr;
  for (let i = 0; i < attempts; i++) {
    try {
      console.log(`📡 Calling AI: ${url.split('/').pop()}`);
      const r = await axios.post(url, payload, { timeout: opts.timeout || 30000 });
      console.log(`✅ AI Response: ${url.split('/').pop()}`);
      return r.data;
    } catch (err) {
      lastErr = err;
      console.warn(`⚠️ AI call failed (attempt ${i + 1}/${attempts}):`, err.message);
      if (i < attempts - 1) {
        await new Promise(r => setTimeout(r, backoffMs * (i + 1)));
      }
    }
  }
  throw lastErr;
}

async function callAiGenerateQuestion(payload) {
  return callWithRetry(`${AI_URL}/generate_question`, payload, { timeout: 30000 }, 2, 300);
}

async function callAiScoreAnswer(payload) {
  return callWithRetry(`${AI_URL}/score_answer`, payload, { timeout: 30000 }, 2, 300);
}

async function callAiProbe(payload) {
  return callWithRetry(`${AI_URL}/probe`, payload, { timeout: 30000 }, 2, 300);
}

async function callAiFinalizeDecision(payload) {
  return callWithRetry(`${AI_URL}/finalize_decision`, payload, { timeout: 30000 }, 2, 300);
}

// ---------------- ROUTES ----------------

// Health check
app.get("/health", (req, res) => {
  res.json({ 
    status: "backend running", 
    ai_service: AI_URL,
    timestamp: new Date().toISOString()
  });
});

// Test route to verify routing works
app.get("/test", (req, res) => {
  res.json({ message: "Server is working!" });
});

// Auth routes
app.post("/auth/signup", async (req, res) => {
  try {
    console.log("📝 Signup request:", req.body.email);
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

app.post("/auth/login", async (req, res) => {
  try {
    console.log("🔐 Login request:", req.body.email);
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

    const aiResponse = await axios.post(`${AI_URL}/parse_resume`, form, {
      headers: { ...form.getHeaders() },
      maxContentLength: Infinity,
      maxBodyLength: Infinity,
      timeout: 30000,
    });

    const parsed = aiResponse.data?.parsed ?? aiResponse.data ?? null;
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
      error: "Failed to parse resume", 
      details: process.env.NODE_ENV === "production" ? undefined : details 
    });
  }
});

// Interview start
app.post("/interview/start", requireAuth, async (req, res) => {
  try {
    console.log("🎬 Starting interview for user:", req.userId);
    const body = req.body || {};
    const userId = req.userId || null;

    // Create session
    const session = await createSessionDB(userId, { from: "frontend" });
    console.log("📝 Created session:", session.sessionId);

    // Build AI payload
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

    // Call AI
    const aiResp = await callAiGenerateQuestion(aiPayload);
    const parsed = aiResp.parsed || {};
    
    const questionText = parsed.question || parsed.questionText || 
      "Tell me about the most technically challenging project on your resume. What specific problem did you solve?";
    
    const qaMetadata = {
      target_project: parsed.target_project,
      technology_focus: parsed.technology_focus,
      red_flags: parsed.red_flags || [],
      confidence: parsed.confidence
    };

    // Create QA record
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
        expectedAnswerType: qaDoc.expectedAnswerType,
        difficulty: qaDoc.difficulty,
        ideal_outline: qaDoc.ideal_outline,
        red_flags: parsed.red_flags
      }
    });
  } catch (err) {
    console.error("❌ Interview start error:", err.message);
    const details = err?.response?.data ?? err?.message ?? String(err);
    return res.status(500).json({ 
      error: "failed_to_start_interview", 
      details: process.env.NODE_ENV === "production" ? undefined : details 
    });
  }
});

// Interview answer - THE MAIN ENDPOINT
app.post("/interview/answer", requireAuth, async (req, res) => {
  try {
    console.log("💬 Processing answer for session:", req.body.sessionId);
    
    const { sessionId, qaId, questionId, questionText, candidateAnswer, candidate_answer } = req.body || {};
    const userId = req.userId || null;
    const finalAnswer = candidateAnswer || candidate_answer || "";

    if (!sessionId) {
      console.error("❌ Missing sessionId");
      return res.status(400).json({ error: "missing sessionId" });
    }
    
    if (!qaId && !questionId) {
      console.error("❌ Missing qaId/questionId");
      return res.status(400).json({ error: "missing qaId or questionId" });
    }

    // Find QA record
    let qaRec = null;
    if (qaId) {
      qaRec = await getQAByQaId(qaId);
    } else {
      qaRec = await QA.findOne({ questionId, sessionId }).lean();
    }
    
    if (!qaRec) {
      console.error("❌ QA record not found:", qaId || questionId);
      return res.status(404).json({ error: "qa_record_not_found" });
    }

    console.log("📝 Found QA record:", qaRec.qaId);

    // Save answer
    await updateQARecordDB(qaRec.qaId, { 
      candidateAnswer: finalAnswer, 
      answeredAt: new Date() 
    });

    // Build question history
    const questionHistory = await buildQuestionHistory(sessionId);
    console.log("📊 Question history length:", questionHistory.length);

    // Score the answer
    const scorePayload = {
      request_id: uuidv4(),
      session_id: sessionId,
      user_id: userId || "anonymous",
      question_text: questionText || qaRec.questionText,
      ideal_outline: qaRec.ideal_outline || "",
      candidate_answer: finalAnswer,
      resume_summary: req.body.resume_summary || "",
      retrieved_chunks: req.body.retrieved_chunks || [],
      question_history: questionHistory,
      token_budget: 1200,
      allow_pii: !!req.body.allow_pii,
      options: { temperature: 0.0 }
    };

    const aiScoreResp = await callAiScoreAnswer(scorePayload);
    const validated = aiScoreResp.validated || {};
    
    console.log("📊 Score received:", validated.overall_score || validated.score);

    // Update QA with score
    const scoreUpdate = {
      gradedBy: "llm",
      score: validated.overall_score || validated.score || 0,
      rubricScores: validated.dimension_scores || validated.rubric_scores || null,
      verdict: validated.verdict || "weak",
      confidence: validated.confidence || 0.5,
      rationale: validated.rationale || "",
      improvement: validated.improvement || validated.follow_up_probe || null,
      red_flags_detected: validated.red_flags_detected || [],
      missing_elements: validated.missing_elements || [],
      needsHumanReview: aiScoreResp.needs_human_review || aiScoreResp.in_gray_zone || false,
      gradedAt: new Date(),
      metadata: { 
        ai_parse_ok: !!aiScoreResp.parse_ok,
        in_gray_zone: aiScoreResp.in_gray_zone || false
      }
    };
    await updateQARecordDB(qaRec.qaId, scoreUpdate);

    // Add to history
    questionHistory.push({
      question: qaRec.questionText,
      questionText: qaRec.questionText,
      answer: finalAnswer,
      score: scoreUpdate.score,
      verdict: scoreUpdate.verdict,
      ideal_outline: qaRec.ideal_outline || ""
    });

    // Check if interview should end
    let nextQuestion = null;
    let ended = false;
    let decisionResult = null;
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
      decisionResult = finalizeResp.result || finalizeResp;
      performanceMetrics = finalizeResp.performance_metrics || null;
      
      const isFinal = finalizeResp.is_final || false;
      const modelDecision = decisionResult?.parsed || decisionResult?.decision || decisionResult;

      if (isFinal && modelDecision && modelDecision.ended) {
        ended = true;
        console.log("🏁 Interview ended:", modelDecision.verdict);
        
        const decisionDoc = await Decision.create({
          decisionId: uuidv4(),
          sessionId,
          decidedBy: "model",
          verdict: modelDecision.verdict,
          confidence: modelDecision.confidence || 0.5,
          reason: modelDecision.reason || "",
          recommended_role: modelDecision.recommended_role || null,
          key_strengths: modelDecision.key_strengths || [],
          critical_weaknesses: modelDecision.critical_weaknesses || [],
          rawModelOutput: decisionResult,
          performanceMetrics: performanceMetrics,
          decidedAt: new Date()
        });
        
        await markSessionCompletedDB(sessionId, { 
          finalDecisionRef: decisionDoc._id,
          performanceMetrics: performanceMetrics
        });
      }
    } catch (e) {
      console.warn("⚠️ Decision check failed:", e.message);
    }

    // Generate next question if not ended
    if (!ended) {
      try {
        const shouldProbe = aiScoreResp.in_gray_zone || 
                           (scoreUpdate.score < 0.60 && scoreUpdate.score >= 0.30);
        
        if (shouldProbe && validated.follow_up_probe) {
          console.log("🔍 Generating probe question");
          const probePayload = {
            request_id: uuidv4(),
            session_id: sessionId,
            user_id: userId || "anonymous",
            weakness_topic: validated.missing_elements?.[0] || "the previous topic",
            prev_question: qaRec.questionText,
            prev_answer: finalAnswer,
            resume_summary: req.body.resume_summary || "",
            retrieved_chunks: req.body.retrieved_chunks || [],
            conversation: req.body.conversation || [],
            token_budget: 600,
            allow_pii: !!req.body.allow_pii
          };
          
          const probeResp = await callAiProbe(probePayload);
          const parsed = probeResp.parsed || {};
          const probeQuestion = parsed.probe_question || validated.follow_up_probe;
          
          const newQa = await createQARecordDB(
            sessionId, 
            probeQuestion, 
            null, 
            parsed.expected_answer_length || "medium", 
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
            parsed.expected_answer_type || "medium", 
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
        console.warn("⚠️ Next question generation failed:", e.message);
        ended = true;
      }
    }

    // Get latest QA data
    const latestQa = await getQAByQaId(qaRec.qaId);
    
    const result = {
      overall_score: latestQa?.score || 0,
      score: latestQa?.score || 0,
      dimension_scores: latestQa?.rubricScores || null,
      rubricScores: latestQa?.rubricScores || null,
      verdict: latestQa?.verdict || "weak",
      confidence: latestQa?.confidence || 0.5,
      rationale: latestQa?.rationale || "",
      red_flags_detected: latestQa?.red_flags_detected || [],
      missing_elements: latestQa?.missing_elements || [],
      improvement: latestQa?.improvement || null,
      follow_up_probe: latestQa?.improvement || null,
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
    console.error("❌ Interview answer error:", err.message);
    console.error(err.stack);
    const details = err?.response?.data ?? err?.message ?? String(err);
    return res.status(500).json({ 
      error: "failed_to_score_answer", 
      details: process.env.NODE_ENV === "production" ? undefined : details 
    });
  }
});

// Interview end
app.post("/interview/end", requireAuth, async (req, res) => {
  try {
    console.log("🏁 Ending interview:", req.body.sessionId);
    const { sessionId } = req.body || {};
    if (!sessionId) return res.status(400).json({ error: "missing sessionId" });

    const s = await getSessionByIdDB(sessionId);
    if (!s) return res.status(404).json({ error: "session_not_found" });

    const recs = await QA.find({ sessionId }).lean();
    const scores = recs.map(r => r.score).filter(v => v !== null && v !== undefined && !isNaN(v));
    const avgScore = scores.length ? (scores.reduce((a, b) => a + b, 0) / scores.length) : null;

    await markSessionCompletedDB(sessionId);
    
    console.log("✅ Interview ended successfully");
    
    return res.json({ 
      ok: true, 
      sessionId, 
      finalScore: avgScore !== null ? Math.round(avgScore * 1000) / 10 : null,
      totalQuestions: recs.length
    });
  } catch (err) {
    console.error("❌ Interview end error:", err.message);
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

// Catch-all 404 handler (must be after all routes)
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
      "POST /interview/answer",
      "POST /interview/end"
    ]
  });
});

// Global error handler
app.use((err, req, res, next) => {
  console.error("❌ Global error:", err.message);
  if (err.code === "LIMIT_FILE_SIZE") {
    return res.status(400).json({ error: "File too large" });
  }
  if (/Only PDF|DOCX|TXT/.test(err.message || "")) {
    return res.status(400).json({ error: err.message });
  }
  res.status(500).json({ error: "internal_server_error" });
});

// ---------------- START SERVER ----------------
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