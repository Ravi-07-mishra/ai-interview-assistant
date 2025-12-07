import mongoose from "mongoose";

const sessionSchema = new mongoose.Schema({
  userId: { type: mongoose.Schema.Types.ObjectId, ref: "User" },
  status: { type: String, enum: ["active", "completed"], default: "active" },
  startedAt: Date,
  endedAt: Date
}, { timestamps: true });

export default mongoose.model("Session", sessionSchema);
