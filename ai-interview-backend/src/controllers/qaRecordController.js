import QARecord from "../models/QARecord.js";

export const saveQA = async (req, res) => {
    try {
        const record = await QARecord.create(req.body);
        res.json(record);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
};
