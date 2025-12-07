import mongoose from "mongoose";

const qaRecordSchema = new mongoose.Schema({
  sessionId: { type: mongoose.Schema.Types.ObjectId, ref: "Session" },
  question: String,
  answer: String,
  aiScore: Number
}, { timestamps: true });

export default mongoose.model("QARecord", qaRecordSchema);
