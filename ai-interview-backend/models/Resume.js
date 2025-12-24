// models/Resume.js
const mongoose = require("mongoose");

const ResumeSchema = new mongoose.Schema({
  userId: { type: mongoose.Schema.Types.ObjectId, ref: "User" },
  sourceUrl: { type: String }, 
  parsed: { type: mongoose.Schema.Types.Mixed }, 
  redactionLog: { type: Array, default: [] },
  rawTextStored: { type: Boolean, default: false }, 
  createdAt: { type: Date, default: Date.now }
}, { timestamps: true });

// 🚀 PERFORMANCE INDEXES
// Quickly get the latest resume for a user
ResumeSchema.index({ userId: 1, createdAt: -1 });

module.exports = mongoose.model("Resume", ResumeSchema);