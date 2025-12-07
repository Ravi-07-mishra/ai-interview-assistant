const QARecord = require("../modells/QARecord");

exports.createQARecord = async (req, res) => {
  try {
    const {
      sessionId,
      question,
      answer,
      aiScore,
      aiFeedback,
      mode,
      usedChunkIds
    } = req.body;

    const record = await QARecord.create({
      sessionId,
      question,
      answer,
      aiScore,
      aiFeedback,
      mode,
      usedChunkIds
    });

    res.status(201).json(record);
  } catch (err) {
    console.error("Create QA record error:", err.message);
    res.status(500).json({ error: "Failed to create QA record" });
  }
};

exports.getQAForSession = async (req, res) => {
  try {
    const { sessionId } = req.params;
    const records = await QARecord.find({ sessionId }).sort({ createdAt: 1 });
    res.json(records);
  } catch (err) {
    console.error("Get QA records error:", err.message);
    res.status(500).json({ error: "Failed to fetch QA records" });
  }
};
