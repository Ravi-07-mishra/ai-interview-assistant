const mongoose = require("mongoose");

const sessionSchema = new mongoose.Schema(
  {
    userId: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true },
    status: {
      type: String,
      enum: ["active", "completed", "cancelled"],
      default: "active"
    },
    metadata: {
      role: String,          // e.g. "SDE-1"
      level: String,         // e.g. "junior"
      position: String
    },
    startedAt: { type: Date, default: Date.now },
    endedAt: { type: Date }
  },
  { timestamps: true }
);

module.exports = mongoose.model("Session", sessionSchema);
