const Document = require("../modells/Document");

exports.createDocument = async (req, res) => {
  try {
    const { sessionId, originalName, text, fileType, source } = req.body;

    const doc = await Document.create({
      userId: req.user._id,
      sessionId,
      originalName,
      text,
      fileType,
      source
    });

    res.status(201).json(doc);
  } catch (err) {
    console.error("Create document error:", err.message);
    res.status(500).json({ error: "Failed to create document" });
  }
};

exports.getDocumentById = async (req, res) => {
  try {
    const doc = await Document.findOne({
      _id: req.params.id,
      userId: req.user._id
    });

    if (!doc) return res.status(404).json({ error: "Document not found" });

    res.json(doc);
  } catch (err) {
    console.error("Get document error:", err.message);
    res.status(500).json({ error: "Failed to fetch document" });
  }
};

exports.getDocumentsForSession = async (req, res) => {
  try {
    const docs = await Document.find({
      userId: req.user._id,
      sessionId: req.params.sessionId
    }).sort({ createdAt: 1 });

    res.json(docs);
  } catch (err) {
    console.error("Get docs for session error:", err.message);
    res.status(500).json({ error: "Failed to fetch documents" });
  }
};
