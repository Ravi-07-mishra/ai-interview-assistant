import Session from "../models/Session.js";

export const startSession = async (req, res) => {
    const session = await Session.create({ userId: req.body.userId });
    res.json(session);
};

export const endSession = async (req, res) => {
    const session = await Session.findByIdAndUpdate(
        req.params.id, 
        { status: "ended", endedAt: new Date() },
        { new: true }
    );
    res.json(session);
};
