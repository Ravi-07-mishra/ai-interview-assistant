import Feedback from "../models/Feedback.js";

export const giveFeedback = async (req, res) => {
    const fb = await Feedback.create(req.body);
    res.json(fb);
};
