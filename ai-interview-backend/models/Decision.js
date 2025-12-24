// models/Decision.js
const mongoose = require("mongoose");

const DecisionSchema = new mongoose.Schema({
  decisionId: { type: String, required: true, unique: true },
  sessionId: { type: String, required: true }, // Linked Session
  
  decidedBy: { type: String }, // "model" | "human" | "system"
  verdict: { type: String, enum: ["hire", "reject", "maybe"] },
  confidence: { type: Number },
  reason: { type: String },
  
  feedback_summary: { type: String }, // Add this field (used in main.py)
  recommended_role: { type: String },
  key_strengths: { type: [String] },
  critical_weaknesses: { type: [String] },
  
  rawModelOutput: { type: mongoose.Schema.Types.Mixed },
  performanceMetrics: { type: mongoose.Schema.Types.Mixed }, // Store final stats here
  decidedAt: { type: Date, default: Date.now }
}, { timestamps: true });

// 🚀 PERFORMANCE INDEXES
DecisionSchema.index({ sessionId: 1 });

module.exports = mongoose.model("Decision", DecisionSchema);