import LLMSample from "../models/LLMSample.js";

export const saveSample = async (req, res) => {
    const sample = await LLMSample.create(req.body);
    res.json(sample);
};
