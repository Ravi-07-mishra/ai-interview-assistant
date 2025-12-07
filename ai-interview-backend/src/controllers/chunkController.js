const Chunk = require("../modells/Chunk");

exports.saveChunks = async (req, res) => {
  try {
    const { documentId, chunks } = req.body;

    if (!documentId || !Array.isArray(chunks)) {
      return res.status(400).json({ error: "documentId and chunks[] required" });
    }

    const payload = chunks.map((c, idx) => ({
      documentId,
      index: c.index ?? idx,
      content: c.content,
      vector: c.vector || []
    }));

    const created = await Chunk.insertMany(payload);
    res.status(201).json(created);
  } catch (err) {
    console.error("Save chunks error:", err.message);
    res.status(500).json({ error: "Failed to save chunks" });
  }
};

exports.getChunksForDocument = async (req, res) => {
  try {
    const { documentId } = req.params;
    const chunks = await Chunk.find({ documentId }).sort({ index: 1 });
    res.json(chunks);
  } catch (err) {
    console.error("Get chunks error:", err.message);
    res.status(500).json({ error: "Failed to fetch chunks" });
  }
};
