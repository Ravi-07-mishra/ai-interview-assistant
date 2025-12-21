// db/operations.js - Final: Pure MongoDB (No Redis)
const { v4: uuidv4 } = require("uuid");
const Session = require("../models/Session");
const QA = require("../models/QA");
const Decision = require("../models/Decision");

// ❌ Removed: Redis Cache & TTL configurations

class DatabaseOperations {
  
  // =========== SESSION OPERATIONS ===========
  
  async createSession(userId = null, metadata = {}) {
    const sessionId = uuidv4();
  
    // ✅ ENFORCE correct round_progress structure
    const defaultRoundProgress = {
      screening: { questions: 0, status: "in_progress" },
      technical: { questions: 0, status: "not_started" },
      behavioral: { questions: 0, status: "not_started" }
    };

    const session = await Session.create({
      sessionId,
      userId,
      metadata: {
        ...metadata,
        current_round: metadata.current_round || "screening",
        round_progress: (
          metadata.round_progress && 
          typeof metadata.round_progress.screening === 'object'
        ) 
        ? metadata.round_progress 
        : defaultRoundProgress
      },
      status: "active",
      qaIds: [],
      events: [],
      startedAt: new Date(),
      violationCount: 0
    });
  
    console.log(`✅ Session created: ${sessionId}`);
    return session.toObject();
  }

  async getSession(sessionId, useCache = true) { // useCache arg kept for compatibility but ignored
    const session = await Session.findOne({ sessionId })
      .select("-events -metadata.referenceFace")
      .lean()
      .exec();
    
    return session;
  }
  
  async getSessionWithDetails(sessionId) {
    return Session.findOne({ sessionId })
      .populate("finalDecisionRef")
      .lean()
      .exec();
  }
  
  async updateSession(sessionId, updates) {
    // ✅ FIX: Handle round_progress updates safely
    const updateOp = { $set: { ...updates, updatedAt: new Date() } };
  
    // If updating round_progress, ensure it's an object
    if (updates['metadata.round_progress']) {
      const rp = updates['metadata.round_progress'];
      
      // Validate structure
      if (typeof rp !== 'object' || 
          typeof rp.screening !== 'object' ||
          typeof rp.technical !== 'object' ||
          typeof rp.behavioral !== 'object') {
        
        console.warn(`⚠️ Invalid round_progress structure, resetting to default`);
        
        updateOp.$set['metadata.round_progress'] = {
          screening: { questions: 0, status: "not_started" },
          technical: { questions: 0, status: "not_started" },
          behavioral: { questions: 0, status: "not_started" }
        };
      }
    }
  
    const session = await Session.findOneAndUpdate(
      { sessionId },
      updateOp,
      { new: true, lean: true, runValidators: true }
    );
  
    return session;
  }

  async markSessionCompleted(sessionId, extras = {}) {
    return Session.findOneAndUpdate(
      { sessionId },
      { 
        $set: { 
          status: "completed", 
          endedAt: new Date(),
          updatedAt: new Date(),
          ...extras 
        } 
      },
      { new: true, lean: true }
    );
  }
  
  async recordViolation(sessionId, reason, action = "warning") {
    try {
      // ✅ CRITICAL FIX: Use findOneAndUpdate with proper MongoDB operators
      const result = await Session.findOneAndUpdate(
        { sessionId },
        {
          $inc: { violationCount: 1 },
          $push: {
            events: {
              id: uuidv4(),
              type: "violation",
              reason,
              at: new Date(),
              action
            }
          },
          $set: { updatedAt: new Date() },
          // ✅ ENSURE round_progress exists and has correct structure if we are creating/upserting
          $setOnInsert: {
            'metadata.round_progress': {
              screening: { questions: 0, status: 'not_started' },
              technical: { questions: 0, status: 'not_started' },
              behavioral: { questions: 0, status: 'not_started' }
            }
          }
        },
        { 
          new: true, 
          upsert: false,
          runValidators: true 
        }
      );

      if (!result) {
        console.error(`❌ Session not found: ${sessionId}`);
        return { session: null, event: null, violationCount: 0 };
      }

      const violationCount = result.violationCount || 0;
      const event = result.events[result.events.length - 1];

      console.log(
        `⚠️ Violation recorded for ${sessionId}: ${reason} (count=${violationCount})`
      );

      return {
        session: result,
        event,
        violationCount
      };

    } catch (err) {
      console.error(`❌ recordViolation failed for ${sessionId}:`, err.message);
      throw err;
    }
  }

  // =========== QA OPERATIONS ===========
  
  async createQARecord(sessionId, questionText, options = {}) {
    const {
      ideal_outline = null,
      expectedAnswerType = "text",
      difficulty = "medium",
      userId = null,
      metadata = {}
    } = options;
    
    const qaId = uuidv4();
    const questionId = uuidv4();
    
    const qa = await QA.create({
      qaId,
      questionId,
      sessionId,
      userId,
      questionText,
      ideal_outline,
      expectedAnswerType,
      difficulty,
      metadata,
      askedAt: new Date()
    });
    
    await Session.updateOne(
      { sessionId },
      { 
        $push: { qaIds: qaId },
        $set: { updatedAt: new Date() }
      }
    );
    
    return qa.toObject();
  }
  
  async updateQARecord(qaId, updates) {
    const qa = await QA.findOneAndUpdate(
      { qaId },
      { $set: updates },
      { new: true, lean: true }
    );
    
    return qa;
  }
  
  async getQA(qaId) {
    return QA.findOne({ qaId }).lean().exec();
  }
  
  async buildQuestionHistory(sessionId, excludeQaId = null, useCache = true) {
    // ❌ Removed cache checks
    
    const query = { sessionId };
    if (excludeQaId) {
      query.qaId = { $ne: excludeQaId };
    }
    
    const qaDocs = await QA.find(query)
      .select("questionText candidateAnswer score verdict ideal_outline metadata expectedAnswerType")
      .sort({ askedAt: 1 })
      .lean()
      .exec();
    
    const history = qaDocs.map(r => {
      let qType = r.metadata?.type;
      
      if (!qType || qType === "conceptual") {
        const text = (r.questionText || "").toLowerCase();
        if (r.expectedAnswerType === "code" || text.includes("function") || text.includes("code")) {
          qType = "coding_challenge";
        } else if (r.metadata?.target_project || text.includes("project")) {
          qType = "project_discussion";
        }
      }
      
      return {
        question: r.questionText,
        questionText: r.questionText,
        answer: r.candidateAnswer || "",
        score: typeof r.score === "number" ? r.score : 0,
        verdict: r.verdict || null,
        ideal_outline: r.ideal_outline || "",
        type: qType || "conceptual",
        target_project: r.metadata?.target_project || null,
        is_probe: r.metadata?.is_probe || false
      };
    });
    
    console.log(`📜 History built for ${sessionId}: ${history.length} items`);
    return history;
  }
  
  async getSessionStatistics(sessionId) {
    // ❌ Removed cache checks
    
    const stats = await QA.getSessionStats(sessionId);
    const result = stats[0] || {
      avgScore: 0,
      totalQuestions: 0,
      answeredQuestions: 0,
      avgAnswerLength: 0
    };
    
    return result;
  }
  
  // =========== DECISION OPERATIONS ===========
  
  async createDecision(sessionId, decisionData) {
    const decisionId = uuidv4();
    
    const decision = await Decision.create({
      decisionId,
      sessionId,
      ...decisionData,
      decidedAt: new Date()
    });
    
    await Session.updateOne(
      { sessionId },
      { $set: { finalDecisionRef: decision._id } }
    );
    
    return decision.toObject();
  }
  
  // =========== ADMIN OPERATIONS ===========
  
  async getSessionWithQAs(sessionId) {
    const session = await Session.findOne({ sessionId })
      .populate("finalDecisionRef")
      .lean()
      .exec();
    
    if (!session) return null;
    
    const qas = await QA.find({ sessionId })
      .select("-metadata.whiteboard_snapshot")
      .lean()
      .exec();
    
    return { session, qas };
  }
  
  async getUserSessions(userId, limit = 10) {
    return Session.find({ userId })
      .select("-events -metadata.referenceFace")
      .sort({ startedAt: -1 })
      .limit(limit)
      .lean()
      .exec();
  }
  
  // =========== BULK OPERATIONS ===========
  
  async bulkUpdateQAs(sessionId, updates) {
    const result = await QA.updateMany(
      { sessionId },
      { $set: updates }
    );
    
    return result;
  }
  
  // =========== CLEANUP OPERATIONS ===========
  
  async cleanupAbandonedSessions() {
    const cutoffDate = new Date(Date.now() - 48 * 60 * 60 * 1000);
    
    const result = await Session.updateMany(
      {
        status: "active",
        updatedAt: { $lt: cutoffDate }
      },
      {
        $set: {
          status: "aborted",
          endedReason: "Session timeout",
          endedAt: new Date()
        }
      }
    );
    
    console.log(`🧹 Cleaned up ${result.modifiedCount} abandoned sessions`);
    return result.modifiedCount;
  }
  
  // =========== CACHE MANAGEMENT (Stubs) ===========
  // Kept as no-ops for backward compatibility if other files still call them
  
  async clearCache(pattern = null) {
    // no-op
  }
  
  getCacheStats() {
    return { hits: 0, misses: 0, keys: 0, connected: false };
  }
}

module.exports = new DatabaseOperations();