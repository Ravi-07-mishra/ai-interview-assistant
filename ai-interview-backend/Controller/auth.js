// Controller/auth.js
const bcrypt = require("bcrypt");
const jwt = require("jsonwebtoken");
const User = require("../models/User");

const SALT_ROUNDS = parseInt(process.env.SALT_ROUNDS || "12", 10);
const JWT_SECRET = process.env.JWT_SECRET || "";
const JWT_EXPIRES = process.env.JWT_EXPIRES || "7d"; // e.g. "1h", "7d"

// fail fast on weak config in production
if (process.env.NODE_ENV === "production") {
  if (!JWT_SECRET || JWT_SECRET.length < 32) {
    console.error("FATAL: JWT_SECRET must be set to a strong secret in production (>=32 chars)");
    process.exit(1);
  }
}

/**
 * signToken(payload) -> token
 * Keep payload minimal: { userId, tokenVersion }
 */
function signToken(payload) {
  return jwt.sign(payload, JWT_SECRET, { expiresIn: JWT_EXPIRES });
}

/**
 * signupUser({ name, email, password })
 * returns created user (without passwordHash)
 */
async function signupUser({ name, email, password }) {
  if (!email || !password) throw new Error("email and password required");

  const emailNorm = String(email).trim().toLowerCase();
  const existing = await User.findOne({ email: emailNorm }).lean();
  if (existing) throw new Error("Email already registered");

  const passwordHash = await bcrypt.hash(password, SALT_ROUNDS);

  const user = await User.create({
    name: name || "",
    email: emailNorm,
    passwordHash,
    role: "user",
  });

  const safe = user.toObject ? user.toObject() : user;
  delete safe.passwordHash;
  return safe;
}

/**
 * loginUser({ email, password })
 * returns { user: safeUser, token }
 */
async function loginUser({ email, password }) {
  if (!email || !password) throw new Error("email and password required");

  const emailNorm = String(email).trim().toLowerCase();
  const user = await User.findOne({ email: emailNorm });
  if (!user) throw new Error("Invalid credentials");

  if (user.disabled) throw new Error("Account disabled");

  const ok = await bcrypt.compare(password, user.passwordHash);
  if (!ok) throw new Error("Invalid credentials");

  // Update lastLogin
  user.lastLogin = new Date();
  await user.save();

  const payload = { userId: String(user._id), tokenVersion: user.tokenVersion || 0 };
  const token = signToken(payload);

  const safe = user.toObject ? user.toObject() : user;
  delete safe.passwordHash;
  return { user: safe, token };
}

/**
 * verifyTokenRaw(token)
 * - pure function: returns decoded payload or throws.
 * - useful when you need to examine token programmatically.
 */
function verifyTokenRaw(token) {
  if (!token) throw new Error("Missing token");
  const decoded = jwt.verify(token, JWT_SECRET); // may throw
  return decoded;
}

/**
 * verifyTokenMiddleware(options = {})
 * Express middleware version.
 * options:
 *   - requireValid (default true): if false, it will just attach userId when available, otherwise 401 on missing/invalid
 *   - checkTokenVersion (default true): verify tokenVersion against DB (stronger revocation)
 */
function verifyTokenMiddleware({ requireValid = true, checkTokenVersion = true } = {}) {
  return async function (req, res, next) {
    try {
      const auth = req.headers.authorization || req.headers.Authorization || "";
      const token = typeof auth === "string" && auth.startsWith("Bearer ") ? auth.split(" ")[1] : null;
      if (!token) {
        if (requireValid) return res.status(401).json({ message: "missing token" });
        req.userId = null;
        return next();
      }

      const decoded = verifyTokenRaw(token);
      // Optionally check tokenVersion in DB (revocation support)
      if (checkTokenVersion) {
        try {
          const user = await User.findById(decoded.userId).select("tokenVersion disabled").lean();
          if (!user) return res.status(401).json({ message: "invalid token" });
          if (user.disabled) return res.status(401).json({ message: "account disabled" });
          const tv = user.tokenVersion || 0;
          if (decoded.tokenVersion !== undefined && decoded.tokenVersion !== tv) {
            return res.status(401).json({ message: "token revoked" });
          }
        } catch (e) {
          console.warn("tokenVersion check error:", e);
        }
      }

      req.userId = decoded.userId || decoded.user_id || decoded.sub;
      req.tokenPayload = decoded;
      return next();
    } catch (err) {
      if (requireValid) return res.status(401).json({ message: "invalid token" });
      req.userId = null;
      return next();
    }
  };
}

// Export both ready middleware and factory
module.exports = {
  signupUser,
  loginUser,
  signToken,
  verifyTokenRaw,
  verifyTokenMiddleware,
  // a ready-to-use middleware (default requireValid = true)
  verifyToken: verifyTokenMiddleware(),
};
