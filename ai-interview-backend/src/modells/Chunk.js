import mongoose from "mongoose";

const chunkSchema = new mongoose.Schema({
  documentId: { type: mongoose.Schema.Types.ObjectId, ref: "Document" },
  content: String,
  vector: [Number]
}, { timestamps: true });

export default mongoose.model("Chunk", chunkSchema);
