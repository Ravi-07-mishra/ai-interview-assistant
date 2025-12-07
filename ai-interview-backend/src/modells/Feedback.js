import mongoose from "mongoose";

const feedbackSchema = new mongoose.Schema({
  sessionId: { type: mongoose.Schema.Types.ObjectId, ref: "Session" },
  qaRecordId: { type: mongoose.Schema.Types.ObjectId, ref: "QARecord" },
  feedback: { type: String, enum: ["up", "down"] }
}, { timestamps: true });

export default mongoose.model("Feedback", feedbackSchema);
