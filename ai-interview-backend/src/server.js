require("dotenv").config();
const express = require("express");
const axios = require("axios");
const cors = require("cors");
const multer = require("multer");
const FormData = require("form-data");
const app = express();

// basic middleware
app.use(express.json());

// --- DEBUG LOGGER: show incoming requests ---
app.use((req, res, next) => {
  console.log(`[${new Date().toISOString()}] Incoming:`, req.method, req.url);
  next();
});

// --- CORS: allow preflight and all origins while dev/debugging ---
app.use(cors());
// app.options("/*", cors());
 // explicitly handle preflight

// Multer memory storage with limits + extension check
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 }, // 10 MB
  fileFilter: (req, file, cb) => {
    const allowed = [".pdf", ".docx", ".txt"];
    const name = (file.originalname || "").toLowerCase();
    if (allowed.some(ext => name.endsWith(ext))) {
      cb(null, true);
    } else {
      cb(new Error("Only PDF, DOCX or TXT files are allowed"));
    }
  },
});

const AI_URL = (process.env.AI_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

// HEALTH
app.get("/health", (req, res) => {
  res.json({ status: "backend running" });
});

// PROCESS RESUME
app.post("/process-resume", upload.single("file"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No file uploaded" });
    }

    const form = new FormData();
    form.append("file", req.file.buffer, {
      filename: req.file.originalname,
      contentType: req.file.mimetype || "application/octet-stream",
      knownLength: req.file.size,
    });

    const aiResponse = await axios.post(`${AI_URL}/parse_resume`, form, {
      headers: {
        ...form.getHeaders(),
      },
      maxContentLength: Infinity,
      maxBodyLength: Infinity,
      timeout: 30000,
    });

    const parsed = aiResponse.data?.parsed ?? aiResponse.data ?? null;
    return res.json({ parsed });

  } catch (err) {
    console.error("Parse Resume Error:", err?.message || err);
    const details = err?.response?.data ?? err?.message ?? String(err);
    return res.status(500).json({
      error: "Failed to parse resume",
      details,
    });
  }
});

// EVALUATE endpoint (forwarding)
app.post("/evaluate", async (req, res) => {
  try {
    const response = await axios.post(`${AI_URL}/evaluate`, req.body, {
      timeout: 15000,
    });
    res.json(response.data);
  } catch (err) {
    console.error("AI call error:", err?.message || err);
    const details = err?.response?.data ?? err?.message ?? String(err);
    res.status(500).json({
      error: "AI service error",
      details,
    });
  }
});

// --- MULTER & GENERAL ERROR HANDLER (returns JSON) ---
app.use((err, req, res, next) => {
  console.error("Global error:", err && err.message ? err.message : err);
  if (err && err.code === "LIMIT_FILE_SIZE") {
    return res.status(400).json({ error: "File too large" });
  }
  if (err && /Only PDF|DOCX|TXT/.test(err.message || "")) {
    return res.status(400).json({ error: err.message });
  }
  res.status(500).json({ error: "internal_server_error", details: err?.message ?? String(err) });
});

// START
const PORT = process.env.PORT || 4000;
app.listen(PORT, () => {
  console.log(`Backend running on port ${PORT}`);
});
