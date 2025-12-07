import Chunk from "../models/Chunk.js";

export const saveChunks = async (req, res) => {
    try {
        const chunks = await Chunk.insertMany(req.body.chunks);
        res.json(chunks);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
};
