const redisCache = require('./redisCache');
const { v4: uuidv4 } = require('uuid');

class SessionLockManager {
  constructor() {
    this.lockTTL = 45;        // seconds (covers longest AI operations)
    this.retryDelay = 300;    // ms
    this.maxRetries = 1;      // fail fast
  }

  _lockKey(sessionId, operation) {
    return `lock:${sessionId}:${operation}`;
  }

  async acquireLock(sessionId, operation = 'operation') {
    const lockKey = this._lockKey(sessionId, operation);
    const lockToken = uuidv4();

    for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
      try {
        const result = await redisCache.client?.set(
          lockKey,
          lockToken,
          'NX',
          'EX',
          this.lockTTL
        );

        if (result === 'OK') {
          console.log(`🔒 Lock acquired: ${sessionId} (${operation}) - token: ${lockToken.slice(0, 8)}`);
          return lockToken;
        }

        const ttl = await redisCache.client?.ttl(lockKey);

        // Fix stale locks without expiry
        if (ttl === -1) {
          console.warn(`⚠️ Fixing stale lock for ${sessionId} (${operation})`);
          await redisCache.client?.del(lockKey);
          continue;
        }

        if (attempt < this.maxRetries) {
          console.log(`⏳ Lock busy: ${sessionId} (${operation}), waiting ${this.retryDelay}ms (TTL: ${ttl}s)`);
          await new Promise(r => setTimeout(r, this.retryDelay));
        }

      } catch (err) {
        console.warn(`⚠️ Lock error (${operation}) attempt ${attempt + 1}:`, err.message);
        if (attempt === this.maxRetries) throw err;
      }
    }

    console.warn(`❌ Lock acquisition failed: ${sessionId} (${operation})`);
    return null;
  }

  async releaseLock(sessionId, operation, lockToken) {
    const lockKey = this._lockKey(sessionId, operation);

    try {
      const currentToken = await redisCache.client?.get(lockKey);

      if (currentToken && String(currentToken) === String(lockToken)) {
        await redisCache.client?.del(lockKey);
        console.log(`🔓 Lock released: ${sessionId} (${operation}) - token: ${lockToken.slice(0, 8)}`);
        return true;
      }

      console.warn(`⚠️ Lock token mismatch/expired: ${sessionId} (${operation})`);
      return false;

    } catch (err) {
      console.error(`❌ Lock release error: ${sessionId} (${operation})`, err.message);
      return false;
    }
  }

  /**
   * Execute function with lock protection
   * @param {string} sessionId - Session identifier
   * @param {string} operation - Operation name (answer, violation, proctor, etc.)
   * @param {Function} fn - Async function to execute
   * @param {Object} opts - Options
   * @param {number} opts.timeout - Max execution time (ms). Defaults based on operation:
   *   - answer: 35000ms (scoring + question generation)
   *   - violation: 15000ms (DB writes + potential termination)
   *   - proctor: 10000ms (face verification)
   *   - default: 20000ms
   */
  async withLock(sessionId, operation, fn, opts = {}) {
    // 🔥 OPERATION-SPECIFIC TIMEOUTS
    const DEFAULT_TIMEOUTS = {
      answer: 35000,      // Longest: AI scoring + question generation
      violation: 15000,   // DB updates + possible decision creation
      proctor: 10000,     // Quick face verification
      hint: 15000,        // AI hint generation
      start: 25000,       // Face registration + question generation
      default: 20000
    };

    const MAX_WAIT = opts.timeout ?? DEFAULT_TIMEOUTS[operation] ?? DEFAULT_TIMEOUTS.default;
    const startTime = Date.now();

    const lockToken = await this.acquireLock(sessionId, operation);
    if (!lockToken) {
      throw new Error(`Failed to acquire lock (${operation}) for session ${sessionId}`);
    }

    try {
      const result = await Promise.race([
        fn(),
        new Promise((_, reject) =>
          setTimeout(
            () => reject(new Error(`Lock operation timeout (${operation}) after ${MAX_WAIT}ms`)), 
            MAX_WAIT
          )
        )
      ]);

      const elapsed = Date.now() - startTime;
      
      // Warn about slow operations (but allow them to complete)
      if (elapsed > 5000) {
        console.warn(`⏱️ Slow ${operation}: ${elapsed}ms (limit: ${MAX_WAIT}ms)`);
      }

      return result;
    } finally {
      await this.releaseLock(sessionId, operation, lockToken);
    }
  }

  // 🔥 Force clear all locks (admin/debug)
  async forceReleaseAll() {
    try {
      const keys = await redisCache.client?.keys('lock:*');
      if (keys?.length) {
        await redisCache.client?.del(...keys);
        console.log(`🧹 Force released ${keys.length} locks`);
        return keys.length;
      }
      return 0;
    } catch (err) {
      console.error('❌ Force release failed:', err.message);
      return 0;
    }
  }

  async isLocked(sessionId, operation) {
    try {
      const key = this._lockKey(sessionId, operation);
      return !!(await redisCache.client?.get(key));
    } catch {
      return false;
    }
  }

  async getLockInfo(sessionId, operation) {
    try {
      const key = this._lockKey(sessionId, operation);
      const [token, ttl] = await Promise.all([
        redisCache.client?.get(key),
        redisCache.client?.ttl(key)
      ]);

      return token
        ? { locked: true, operation, ttl, token: token.slice(0, 8) }
        : { locked: false };
    } catch (err) {
      return { locked: false, error: err.message };
    }
  }
}

module.exports = new SessionLockManager();