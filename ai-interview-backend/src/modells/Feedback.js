const mongoose = require("mongoose");

const feedbackSchema = new mongoose.Schema(
  {
    sessionId: { type: mongoose.Schema.Types.ObjectId, ref: "Session", required: true },
    qaRecordId: { type: mongoose.Schema.Types.ObjectId, ref: "QARecord", required: true },
    feedback: { type: String, enum: ["up", "down"], required: true }
  },
  { timestamps: true }
);

module.exports = mongoose.model("Feedback", feedbackSchema);
