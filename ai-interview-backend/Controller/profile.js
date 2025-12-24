const User = require("../models/User");
const Session = require("../models/Session");
const QA = require("../models/QA");
const Decision = require("../models/Decision");

async function getProfileDashboard(req, res) {
  try {
    const userId = req.userId;

    // ================= USER =================
    const user = await User.findById(userId)
      .select("name email role createdAt lastLogin")
      .lean();

    // ================= SESSIONS =================
    const sessions = await Session.find({ userId })
      .sort({ startedAt: -1 })
      .lean();

    const totalInterviews = sessions.length;

    // ================= QAs (Scores) =================
    const qas = await QA.find({ userId }).lean();
    const scores = qas
      .map(q => q.score)
      .filter(s => typeof s === "number");

    const averageScore =
      scores.length > 0
        ? Math.round((scores.reduce((a, b) => a + b, 0) / scores.length) * 100) / 100
        : 0;

    // ================= INTERVIEW HISTORY =================
    const interviewHistory = [];

    for (const session of sessions) {
      const sessionQAs = qas.filter(
        q => q.sessionId === session.sessionId
      );

      const sessionScores = sessionQAs
        .map(q => q.score)
        .filter(s => typeof s === "number");

      const sessionAvg =
        sessionScores.length > 0
          ? Math.round(
              (sessionScores.reduce((a, b) => a + b, 0) / sessionScores.length) * 100
            ) / 100
          : null;

      const decision = await Decision.findOne({
        sessionId: session.sessionId
      }).lean();

      interviewHistory.push({
        sessionId: session.sessionId,
        date: session.startedAt,
        averageScore: sessionAvg,
        verdict: decision?.verdict || "completed",
        feedback:
          decision?.reason ||
          sessionQAs[sessionQAs.length - 1]?.improvement ||
          null
      });
    }

   return res.json({
  user,
  stats: {
    totalInterviews,
    averageScore
  },
  pastSessions: interviewHistory.map(i => ({
    sessionId: i.sessionId,
    startedAt: i.date,
    status: i.verdict,
    qas: [{
      score: i.averageScore,
      feedback: i.feedback
    }]
  }))
});
  } catch (err) {
    console.error("Profile dashboard error:", err);
    return res.status(500).json({ error: "failed_to_load_profile_dashboard" });
  }
}

module.exports = { getProfileDashboard };
