import Document from "../models/Document.js";

export const saveDocument = async (req, res) => {
    try {
        const doc = await Document.create(req.body);
        res.json(doc);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
};
