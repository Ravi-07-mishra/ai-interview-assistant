import mongoose from "mongoose";

const userSchema = new mongoose.Schema({
  name: String,
  email: { type: String, unique: true },
  password: String,
  consent: {
    analytics: { type: Boolean, default: true },
    dataUsage: { type: Boolean, default: true }
  }
}, { timestamps: true });

export default mongoose.model("User", userSchema);
