// models/Session.js
const mongoose = require("mongoose");

const SessionSchema = new mongoose.Schema({
  sessionId: { type: String, required: true, unique: true }, // Index handled by unique: true
  userId: { type: mongoose.Schema.Types.ObjectId, ref: "User" },
  status: { type: String, enum: ["active", "completed", "abandoned"], default: "active" },
  
  // Timestamps
  startedAt: { type: Date, default: Date.now },
  endedAt: { type: Date },
  
  // Metadata & Refs
  metadata: { type: mongoose.Schema.Types.Mixed },
  resumeRef: { type: mongoose.Schema.Types.ObjectId, ref: "Resume" },
  finalDecisionRef: { type: mongoose.Schema.Types.ObjectId, ref: "Decision" },
  
  // Violation tracking (Optional, but good for quick access)
  violationCount: { type: Number, default: 0 },
  events: { type: Array, default: [] } 
}, { timestamps: true });

// 🚀 PERFORMANCE INDEXES
// 1. Quickly find a user's active session
SessionSchema.index({ userId: 1, status: 1 });

module.exports = mongoose.model("Session", SessionSchema);