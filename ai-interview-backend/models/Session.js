// models/Session.js
const mongoose = require("mongoose");

const SessionSchema = new mongoose.Schema({
  sessionId: { type: String, required: true, unique: true, index: true },
  userId: { type: mongoose.Schema.Types.ObjectId, ref: "User", index: true },
  status: { type: String, enum: ["active","completed","abandoned"], default: "active" },
  startedAt: { type: Date, default: Date.now },
  endedAt: { type: Date },
  metadata: { type: mongoose.Schema.Types.Mixed },
  resumeRef: { type: mongoose.Schema.Types.ObjectId, ref: "Resume" },
  finalDecisionRef: { type: mongoose.Schema.Types.ObjectId, ref: "Decision" }
}, { timestamps: true });

module.exports = mongoose.model("Session", SessionSchema);
