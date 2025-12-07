const express = require("express");
const router = express.Router();

const documentController = require("../controllers/documentController");
const auth = require("../middleware/authMiddleware");

router.post("/", auth, documentController.createDocument);
router.get("/:id", auth, documentController.getDocumentById);
router.get("/session/:sessionId", auth, documentController.getDocumentsForSession);

module.exports = router;
