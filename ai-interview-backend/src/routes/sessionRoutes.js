const express = require("express");
const router = express.Router();

const sessionController = require("../controllers/sessionController");
const auth = require("../middleware/authMiddleware");

router.post("/", auth, sessionController.createSession);
router.patch("/:id/end", auth, sessionController.endSession);
router.get("/:id", auth, sessionController.getSession);

module.exports = router;
