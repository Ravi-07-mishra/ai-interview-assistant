const mongoose = require("mongoose");

const qaRecordSchema = new mongoose.Schema(
  {
    sessionId: { type: mongoose.Schema.Types.ObjectId, ref: "Session", required: true },
    question: { type: String, required: true },
    answer: { type: String, required: true },          // user answer
    aiScore: { type: Number, min: 0, max: 10 },
    aiFeedback: { type: String },
    mode: { type: String, default: "default" },        // for future strategy modes
    usedChunkIds: [{ type: mongoose.Schema.Types.ObjectId, ref: "Chunk" }]
  },
  { timestamps: true }
);

module.exports = mongoose.model("QARecord", qaRecordSchema);
