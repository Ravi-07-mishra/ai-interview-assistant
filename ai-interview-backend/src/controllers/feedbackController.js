const Feedback = require("../modells/Feedback");

exports.createFeedback = async (req, res) => {
  try {
    const { sessionId, qaRecordId, feedback } = req.body;

    const fb = await Feedback.create({
      sessionId,
      qaRecordId,
      feedback
    });

    res.status(201).json(fb);
  } catch (err) {
    console.error("Create feedback error:", err.message);
    res.status(500).json({ error: "Failed to create feedback" });
  }
};
