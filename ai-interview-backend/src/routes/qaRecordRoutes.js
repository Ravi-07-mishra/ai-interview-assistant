import express from "express";
import { saveQA } from "../controllers/qaRecordController.js";

const router = express.Router();

router.post("/save", saveQA);

export default router;
