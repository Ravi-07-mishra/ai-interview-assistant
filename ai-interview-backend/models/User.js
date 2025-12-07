// models/User.js
const mongoose = require("mongoose");

const UserSchema = new mongoose.Schema({
  name: { type: String, trim: true },
  email: { type: String, required: true, unique: true, index: true, lowercase: true, trim: true },
  passwordHash: { type: String, required: true },
  role: { type: String, default: "user" }, // e.g., admin|user
  tokenVersion: { type: Number, default: 0 }, // for refresh token invalidation
  profile: { type: mongoose.Schema.Types.Mixed }, // optional extra info
  createdAt: { type: Date, default: Date.now },
  lastLogin: { type: Date },
  disabled: { type: Boolean, default: false }
}, { timestamps: true });

module.exports = mongoose.model("User", UserSchema);
