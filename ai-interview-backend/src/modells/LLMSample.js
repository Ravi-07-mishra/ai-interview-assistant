import mongoose from "mongoose";

const llmSampleSchema = new mongoose.Schema({
  prompt: String,
  rawOutput: String,
  meta: Object
}, { timestamps: true });

export default mongoose.model("LLMSample", llmSampleSchema);
