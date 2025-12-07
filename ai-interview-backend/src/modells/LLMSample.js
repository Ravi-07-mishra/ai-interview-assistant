const mongoose = require("mongoose");

const llmSampleSchema = new mongoose.Schema(
  {
    sessionId: { type: mongoose.Schema.Types.ObjectId, ref: "Session" },
    prompt: { type: String, required: true },
    rawOutput: { type: String, required: true },
    meta: { type: Object, default: {} }
  },
  { timestamps: true }
);

module.exports = mongoose.model("LLMSample", llmSampleSchema);
