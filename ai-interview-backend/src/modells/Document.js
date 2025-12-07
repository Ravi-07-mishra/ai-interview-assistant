import mongoose from "mongoose";

const documentSchema = new mongoose.Schema({
  userId: { type: mongoose.Schema.Types.ObjectId, ref: "User" },
  sessionId: { type: mongoose.Schema.Types.ObjectId, ref: "Session" },
  originalName: String,
  text: String,            // parsed or redacted text
  fileType: String
}, { timestamps: true });

export default mongoose.model("Document", documentSchema);
