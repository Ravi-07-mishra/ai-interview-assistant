// models/QA.js
const mongoose = require("mongoose");

const QASchema = new mongoose.Schema({
  qaId: { type: String, required: true, unique: true, index: true },
  sessionId: { type: String, index: true }, // sessionId string (matches session.sessionId)
  userId: { type: mongoose.Schema.Types.ObjectId, ref: "User", index: true },
  questionId: { type: String },
  questionText: { type: String },
  ideal_outline: { type: String },
  expectedAnswerType: { type: String },
  difficulty: { type: String },
  askedAt: { type: Date, default: Date.now },
  answeredAt: { type: Date },
  candidateAnswer: { type: String }, // consider redacting PII or storing pointer to S3
  gradedBy: { type: String },
  score: { type: Number, min: 0, max: 1 },
  rubricScores: { type: mongoose.Schema.Types.Mixed },
  confidence: { type: Number, min: 0, max: 1 },
  rationale: { type: String },
  improvement: { type: String },
  needsHumanReview: { type: Boolean, default: false },
  aiRaw: { type: mongoose.Schema.Types.Mixed }, // optional
  metadata: { type: mongoose.Schema.Types.Mixed }
}, { timestamps: true });

module.exports = mongoose.model("QA", QASchema);
