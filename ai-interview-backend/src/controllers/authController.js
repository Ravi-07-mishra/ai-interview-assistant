const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");
const User = require("../modells/User");

const generateToken = (userId) =>
  jwt.sign({ id: userId }, process.env.JWT_SECRET, { expiresIn: "7d" });

exports.register = async (req, res) => {
  try {
    const { name, email, password, consent } = req.body;

    const existing = await User.findOne({ email });
    if (existing) {
      return res.status(400).json({ error: "Email already registered" });
    }

    const hash = await bcrypt.hash(password, 10);

    const user = await User.create({
      name,
      email,
      password: hash,
      consent
    });

    const token = generateToken(user._id);

    res.status(201).json({
      token,
      user: {
        id: user._id,
        name: user.name,
        email: user.email,
        consent: user.consent
      }
    });
  } catch (err) {
    console.error("Register error:", err.message);
    res.status(500).json({ error: "Registration failed" });
  }
};

exports.login = async (req, res) => {
  try {
    const { email, password } = req.body;

    const user = await User.findOne({ email });
    if (!user) return res.status(400).json({ error: "Invalid credentials" });

    const match = await bcrypt.compare(password, user.password);
    if (!match) return res.status(400).json({ error: "Invalid credentials" });

    const token = generateToken(user._id);

    res.json({
      token,
      user: {
        id: user._id,
        name: user.name,
        email: user.email,
        consent: user.consent
      }
    });
  } catch (err) {
    console.error("Login error:", err.message);
    res.status(500).json({ error: "Login failed" });
  }
};

exports.me = async (req, res) => {
  res.json(req.user);
};


