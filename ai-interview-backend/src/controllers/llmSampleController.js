const LLMSample = require("../modells/LLMSample");

exports.createLLMSample = async (req, res) => {
  try {
    const { sessionId, prompt, rawOutput, meta } = req.body;

    const sample = await LLMSample.create({
      sessionId,
      prompt,
      rawOutput,
      meta
    });

    res.status(201).json(sample);
  } catch (err) {
    console.error("Create LLM sample error:", err.message);
    res.status(500).json({ error: "Failed to create LLM sample" });
  }
};
