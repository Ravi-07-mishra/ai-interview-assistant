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
    // ================= INTERVIEW HISTORY (ROUND-WISE) =================
const interviewHistory = [];

for (const session of sessions) {
  const sessionQAs = qas.filter(q => q.sessionId === session.sessionId);

  const rounds = {
    screening: [],
    technical: [],
    behavioral: []
  };
  // group QAs by round
  sessionQAs.forEach(q => {
    const round = q.metadata?.round;
    if (round && rounds[round]) {
      rounds[round].push(q);
    }
  });
  // helper to compute avg + feedback
  const summarizeRound = (qas) => {
    if (!qas.length) return null;

    const scores = qas.map(q => q.score).filter(s => typeof s === "number");
    const avgScore = scores.length
      ? Math.round((scores.reduce((a, b) => a + b, 0) / scores.length) * 100) / 100
      : null;

    const lastQA = qas[qas.length - 1];

    return {
      averageScore: avgScore,
      feedback: lastQA?.improvement || lastQA?.rationale || null
    };
  };

  interviewHistory.push({
    sessionId: session.sessionId,
    date: session.startedAt,
    rounds: {
      screening: summarizeRound(rounds.screening),
      technical: summarizeRound(rounds.technical),
      behavioral: summarizeRound(rounds.behavioral)
    }
  });
}

  return res.json({
  user,
  stats: {
    totalInterviews,
    averageScore
  },
  interviewHistory
});
  } catch (err) {
    console.error("Profile dashboard error:", err);
    return res.status(500).json({ error: "failed_to_load_profile_dashboard" });
  }
}

module.exports = { getProfileDashboard };
