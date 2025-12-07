const mongoose = require("mongoose");

const chunkSchema = new mongoose.Schema(
  {
    documentId: { type: mongoose.Schema.Types.ObjectId, ref: "Document", required: true },
    index: { type: Number },     // order of chunk in doc
    content: { type: String, required: true },
    vector: { type: [Number], default: [] }  // embedding
  },
  { timestamps: true }
);

module.exports = mongoose.model("Chunk", chunkSchema);
