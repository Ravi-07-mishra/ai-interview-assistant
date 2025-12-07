const express = require("express");
const router = express.Router();

const chunkController = require("../controllers/chunkController");
const auth = require("../middleware/authMiddleware");

router.post("/", auth, chunkController.saveChunks);
router.get("/document/:documentId", auth, chunkController.getChunksForDocument);

module.exports = router;
