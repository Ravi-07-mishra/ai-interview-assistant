import express from "express";
import { giveFeedback } from "../controllers/feedbackController.js";

const router = express.Router();

router.post("/", giveFeedback);

export default router;
