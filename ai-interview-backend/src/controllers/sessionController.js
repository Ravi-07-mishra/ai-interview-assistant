const Session = require("../modells/sessions");

exports.createSession = async (req, res) => {
  try {
    const { metadata } = req.body;

    const session = await Session.create({
      userId: req.user._id,
      metadata,
      startedAt: new Date()
    });

    res.status(201).json(session);
  } catch (err) {
    console.error("Create session error:", err.message);
    res.status(500).json({ error: "Failed to create session" });
  }
};

exports.endSession = async (req, res) => {
  try {
    const session = await Session.findOneAndUpdate(
      { _id: req.params.id, userId: req.user._id },
      { status: "completed", endedAt: new Date() },
      { new: true }
    );

    if (!session) return res.status(404).json({ error: "Session not found" });

    res.json(session);
  } catch (err) {
    console.error("End session error:", err.message);
    res.status(500).json({ error: "Failed to end session" });
  }
};

exports.getSession = async (req, res) => {
  try {
    const session = await Session.findOne({
      _id: req.params.id,
      userId: req.user._id
    });
    if (!session) return res.status(404).json({ error: "Session not found" });
    res.json(session);
  } catch (err) {
    console.error("Get session error:", err.message);
    res.status(500).json({ error: "Failed to fetch session" });
  }
};

