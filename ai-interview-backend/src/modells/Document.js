const mongoose = require("mongoose");

const documentSchema = new mongoose.Schema(
  {
    userId: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true },
    sessionId: { type: mongoose.Schema.Types.ObjectId, ref: "Session" },
    originalName: String,
    text: { type: String },          //  parsed resume text
    fileType: String,
    source: { type: String, default: "resume" }
  },
  { timestamps: true }
);

module.exports = mongoose.model("Document", documentSchema);
