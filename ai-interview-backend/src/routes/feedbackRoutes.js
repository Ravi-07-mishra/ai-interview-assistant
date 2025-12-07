const express = require("express");
const router = express.Router();

const feedbackController = require("../controllers/feedbackController");
const auth = require("../middleware/authMiddleware");

router.post("/", auth, feedbackController.createFeedback);

module.exports = router;
