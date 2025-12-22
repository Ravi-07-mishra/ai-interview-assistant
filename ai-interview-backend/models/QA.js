// models/QA.js
const mongoose = require("mongoose");

const QASchema = new mongoose.Schema({
  qaId: { type: String, required: true, unique: true }, // Index handled by unique: true
  sessionId: { type: String, required: true },
  userId: { type: mongoose.Schema.Types.ObjectId, ref: "User" },
  
  // Core Question Data
  questionId: { type: String },
  questionText: { type: String },
  ideal_outline: { type: String },
  expectedAnswerType: { type: String },
  difficulty: { type: String },
  
  // Timing (Indexed for sorting)
  askedAt: { type: Date, default: Date.now },
  answeredAt: { type: Date },
  
  // Answer Data
  candidateAnswer: { type: String }, 
  
  // Scoring
  gradedBy: { type: String },
  score: { type: Number, min: 0, max: 1 },
  rubricScores: { type: mongoose.Schema.Types.Mixed },
  confidence: { type: Number, min: 0, max: 1 },
  rationale: { type: String },
  improvement: { type: String },
  
  // Flags & Metadata
  needsHumanReview: { type: Boolean, default: false },
  aiRaw: { type: mongoose.Schema.Types.Mixed }, 
  metadata: { type: mongoose.Schema.Types.Mixed }
}, { timestamps: true });

// 🚀 PERFORMANCE INDEXES
// 1. Critical for buildQuestionHistory(): Fetch by session, sorted by time
QASchema.index({ sessionId: 1, askedAt: 1 });

// 2. Useful for "Get all my answers" dashboard queries
QASchema.index({ userId: 1, askedAt: -1 });

module.exports = mongoose.model("QA", QASchema);