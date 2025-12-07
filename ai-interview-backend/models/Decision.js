// models/Decision.js
const mongoose = require("mongoose");

const DecisionSchema = new mongoose.Schema({
  decisionId: { type: String, required: true, unique: true, index: true },
  sessionId: { type: String, index: true },
  decidedBy: { type: String }, // "model" | "human"
  verdict: { type: String, enum: ["hire","reject","maybe"] },
  confidence: { type: Number },
  reason: { type: String },
  recommended_role: { type: String },
  rawModelOutput: { type: mongoose.Schema.Types.Mixed },
  decidedAt: { type: Date, default: Date.now }
}, { timestamps: true });

module.exports = mongoose.model("Decision", DecisionSchema);
