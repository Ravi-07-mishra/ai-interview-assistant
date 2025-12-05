// ai-interview-backend/src/server.js
require("dotenv").config();
const express = require("express");
const axios = require("axios");
const cors = require("cors");

const app = express();
app.use(express.json());
app.use(cors());

const AI_URL = process.env.AI_URL || "http://127.0.0.1:8000";

// health
app.get("/health", (req, res) => {
  res.json({ status: "backend running" });
});

// proxy evaluate to AI service
app.post("/evaluate", async (req, res) => {
  try {
    const response = await axios.post(`${AI_URL}/evaluate`, req.body, { timeout: 15000 });
    return res.json(response.data);
  } catch (err) {
    console.error("AI call error:", err?.message || err);
    return res.status(500).json({ error: "AI service error", details: err?.message || String(err) });
  }
});

const PORT = process.env.PORT || 4000;
app.listen(PORT, () => console.log(`Backend running on port ${PORT}`));
