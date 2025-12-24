// middleware/requestDedup.js - Revised: In-Memory (No Redis)

/**
 * Request deduplication middleware
 * Prevents duplicate requests from being processed within a time window
 * Uses local memory (Map) instead of Redis.
 * * Frontend should include:
 * - X-Request-ID header (UUID)
 * - X-Idempotency-Key header (optional)
 */
class RequestDeduplication {
  constructor() {
    this.windowSeconds = 30; // 30 second deduplication window
    // Map stores: requestId -> { timestamp, status, path, response? }
    this.requestCache = new Map();
    
    // Periodically clean up old entries (every 60 seconds)
    setInterval(() => this.cleanupExpired(), 60000);
  }

  /**
   * Middleware to check for duplicate requests
   */
  middleware() {
    return async (req, res, next) => {
      // Only apply to state-changing operations
      if (req.method === 'GET' || req.method === 'HEAD') {
        return next();
      }

      const requestId = req.headers['x-request-id'] || req.headers['x-idempotency-key'];
      
      // If no request ID provided, allow request but warn
      if (!requestId) {
        // Optional: console.warn(`⚠️ No request ID for ${req.method} ${req.path}`);
        return next();
      }

      // Check if this request was already processed or is processing
      const cached = this.requestCache.get(requestId);
      
      if (cached) {
        console.log(`🔁 Duplicate request blocked: ${requestId.slice(0, 8)} - ${req.path}`);
        
        // If we have a saved response, return it (idempotency)
        if (cached.status === 'completed' && cached.response) {
            return res.json(cached.response);
        }

        // Otherwise, it's still processing or failed
        return res.status(409).json({
          error: 'duplicate_request',
          message: 'This request is already being processed or was processed recently.',
          request_id: requestId
        });
      }

      // Mark request as being processed
      this.markProcessing(requestId, req.path);

      // Hook into res.json to cache the result when done
      const originalJson = res.json.bind(res);
      
      res.json = (body) => {
        // Cache successful response so future duplicates get the same result
        if (res.statusCode >= 200 && res.statusCode < 300) {
          this.cacheResponse(requestId, req.path, body);
        } else {
          // If it failed, remove from cache so user can retry
          this.requestCache.delete(requestId);
        }
        return originalJson(body);
      };

      next();
    };
  }

  /**
   * Mark request as being processed
   */
  markProcessing(requestId, path) {
    this.requestCache.set(requestId, {
      status: 'processing',
      path,
      timestamp: Date.now()
    });
  }

  /**
   * Cache successful response
   */
  cacheResponse(requestId, path, response) {
    this.requestCache.set(requestId, {
      status: 'completed',
      path,
      response,
      timestamp: Date.now()
    });
  }

  /**
   * Clean up expired keys to prevent memory leaks
   */
  cleanupExpired() {
    const now = Date.now();
    const expiry = this.windowSeconds * 1000;
    
    for (const [key, value] of this.requestCache.entries()) {
      if (now - value.timestamp > expiry) {
        this.requestCache.delete(key);
      }
    }
  }

  /**
   * Manual cleanup (for testing)
   */
  cleanup() {
    this.requestCache.clear();
  }
}

module.exports = new RequestDeduplication();