import express from "express";
import { startSession, endSession } from "../controllers/sessionController.js";

const router = express.Router();

router.post("/start", startSession);
router.post("/end/:id", endSession);

export default router;
