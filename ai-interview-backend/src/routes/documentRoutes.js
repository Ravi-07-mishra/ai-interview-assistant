import express from "express";
import { saveDocument } from "../controllers/documentController.js";

const router = express.Router();

router.post("/upload", saveDocument);

export default router;
