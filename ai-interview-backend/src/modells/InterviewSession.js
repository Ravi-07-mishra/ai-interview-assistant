import mongoose from "mongoose";

const interviewSessionSchema = new mongoose.Schema({
  userId: { type: mongoose.Schema.Types.ObjectId, ref: "User" },
  questions: Array,
  answers: Array,
  feedback: String,
  score: Number,
  createdAt: { type: Date, default: Date.now }
});

export default mongoose.model("InterviewSession", interviewSessionSchema);
