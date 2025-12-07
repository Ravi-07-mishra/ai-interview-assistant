const express = require("express");
const router = express.Router();

const qaRecordController = require("../controllers/qaRecordController");
const auth = require("../middleware/authMiddleware");

router.post("/", auth, qaRecordController.createQARecord);
router.get("/session/:sessionId", auth, qaRecordController.getQAForSession);

module.exports = router;
