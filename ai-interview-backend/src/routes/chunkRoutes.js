import express from "express";
import { saveChunks } from "../controllers/chunkController.js";

const router = express.Router();

router.post("/save", saveChunks);

export default router;
