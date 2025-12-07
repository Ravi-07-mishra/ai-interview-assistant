import express from "express";
import { saveSample } from "../controllers/llmSampleController.js";

const router = express.Router();

router.post("/save", saveSample);

export default router;
