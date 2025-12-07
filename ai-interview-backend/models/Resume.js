// models/Resume.js
const mongoose = require("mongoose");

const ResumeSchema = new mongoose.Schema({
  userId: { type: mongoose.Schema.Types.ObjectId, ref: "User", index: true, required: false },
  sourceUrl: { type: String }, // s3 url or pointer
  parsed: { type: mongoose.Schema.Types.Mixed }, // parsed JSON
  redactionLog: { type: Array, default: [] },
  rawTextStored: { type: Boolean, default: false }, // whether raw text recorded in DB (consider false + S3)
  createdAt: { type: Date, default: Date.now }
}, { timestamps: true });

module.exports = mongoose.model("Resume", ResumeSchema);
