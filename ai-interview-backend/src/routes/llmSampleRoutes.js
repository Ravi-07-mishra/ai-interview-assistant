const express = require("express");
const router = express.Router();

const llmSampleController = require("../controllers/llmSampleController");
const auth = require("../middleware/authMiddleware");

router.post("/", auth, llmSampleController.createLLMSample);

module.exports = router;
