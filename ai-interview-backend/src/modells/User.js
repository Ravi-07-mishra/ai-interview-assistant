const mongoose = require("mongoose");

const consentSchema = new mongoose.Schema(
  {
    analytics: { type: Boolean, default: true },
    dataUsage: { type: Boolean, default: true }
  },
  { _id: false }
);

const userSchema = new mongoose.Schema(
  {
    name: { type: String, trim: true },
    email: { type: String, unique: true, required: true, lowercase: true },
    password: { type: String, required: true },
    consent: { type: consentSchema, default: () => ({}) }
  },
  { timestamps: true }
);

module.exports = mongoose.model("User", userSchema);
