
const redis = require('redis');

class RedisCache {
  constructor() {
    this.client = null;
    this.isConnected = false;
    this.stats = { hits: 0, misses: 0, keys: 0 };
  }

  async connect() {
    if (this.isConnected && this.client?.isOpen) {
      console.log('⚡ Redis already connected, skipping reconnection');
      return;
    }

    try {
      this.client = redis.createClient({
        url: process.env.REDIS_URL || 'redis://localhost:6379',
        socket: {
          reconnectStrategy: (retries) => {
            if (retries > 10) {
              console.error('❌ Redis max reconnection attempts reached');
              return new Error('Max reconnection attempts reached');
            }
            return Math.min(retries * 100, 3000);
          }
        }
      });

      this.client.on('error', (err) => {
        console.error('❌ Redis Client Error:', err.message);
        this.isConnected = false;
      });

      this.client.on('connect', () => {
        console.log('⚡ Redis connected');
        this.isConnected = true;
      });

      this.client.on('disconnect', () => {
        console.warn('⚠️ Redis disconnected');
        this.isConnected = false;
      });

      this.client.on('reconnecting', () => {
        console.log('🔄 Redis reconnecting...');
      });

      await this.client.connect();
      
    } catch (err) {
      console.error('❌ Redis connection failed:', err.message);
      this.isConnected = false;
    }
  }

  async get(key) {
    if (!this.isConnected || !this.client?.isOpen) {
      this.stats.misses++;
      return null;
    }

    try {
      const value = await this.client.get(key);
      if (value) {
        this.stats.hits++;
        return JSON.parse(value);
      }
      this.stats.misses++;
      return null;
    } catch (err) {
      console.warn(`⚠️ Redis GET error for key ${key}:`, err.message);
      this.stats.misses++;
      return null;
    }
  }

  async set(key, value, ttl = 600) {
    if (!this.isConnected || !this.client?.isOpen) {
      return false;
    }

    try {
      await this.client.setEx(key, ttl, JSON.stringify(value));
      this.stats.keys++;
      return true;
    } catch (err) {
      console.warn(`⚠️ Redis SET error for key ${key}:`, err.message);
      return false;
    }
  }

  /**
   * Set key only if it doesn't exist (for locks)
   * Returns true if set successfully, false if key already exists
   */
  async setNX(key, value, ttl = 10) {
    if (!this.isConnected || !this.client?.isOpen) {
      return false;
    }

    try {
      // NX means "only set if not exists"
      const result = await this.client.set(key, value, {
        NX: true,
        EX: ttl
      });
      
      // Returns 'OK' if successful, null if key already exists
      return result === 'OK';
    } catch (err) {
      console.warn(`⚠️ Redis SETNX error for key ${key}:`, err.message);
      return false;
    }
  }

  /**
   * Get and delete atomically (for consuming one-time tokens)
   */
  async getAndDelete(key) {
    if (!this.isConnected || !this.client?.isOpen) {
      return null;
    }

    try {
      const value = await this.client.get(key);
      if (value) {
        await this.client.del(key);
        return JSON.parse(value);
      }
      return null;
    } catch (err) {
      console.warn(`⚠️ Redis GETDEL error for key ${key}:`, err.message);
      return null;
    }
  }

  async del(key) {
    if (!this.isConnected || !this.client?.isOpen) {
      return false;
    }

    try {
      await this.client.del(key);
      return true;
    } catch (err) {
      console.warn(`⚠️ Redis DEL error for key ${key}:`, err.message);
      return false;
    }
  }

  /**
   * Delete multiple keys by pattern
   */
  async flushPattern(pattern) {
    if (!this.isConnected || !this.client?.isOpen) {
      return 0;
    }

    try {
      const keys = await this.client.keys(pattern);
      if (keys.length > 0) {
        await this.client.del(keys);
      }
      return keys.length;
    } catch (err) {
      console.warn(`⚠️ Redis FLUSH error for pattern ${pattern}:`, err.message);
      return 0;
    }
  }

  /**
   * Increment a counter atomically
   */
  async incr(key, ttl = 3600) {
    if (!this.isConnected || !this.client?.isOpen) {
      return null;
    }

    try {
      const value = await this.client.incr(key);
      
      // Set expiration only if this is the first increment
      if (value === 1) {
        await this.client.expire(key, ttl);
      }
      
      return value;
    } catch (err) {
      console.warn(`⚠️ Redis INCR error for key ${key}:`, err.message);
      return null;
    }
  }

  /**
   * Get time-to-live for a key
   */
  async ttl(key) {
    if (!this.isConnected || !this.client?.isOpen) {
      return -1;
    }

    try {
      return await this.client.ttl(key);
    } catch (err) {
      console.warn(`⚠️ Redis TTL error for key ${key}:`, err.message);
      return -1;
    }
  }

  /**
   * Check if key exists
   */
  async exists(key) {
    if (!this.isConnected || !this.client?.isOpen) {
      return false;
    }

    try {
      const result = await this.client.exists(key);
      return result === 1;
    } catch (err) {
      console.warn(`⚠️ Redis EXISTS error for key ${key}:`, err.message);
      return false;
    }
  }

  getStats() {
    return {
      ...this.stats,
      connected: this.isConnected
    };
  }

  async disconnect() {
    if (this.client?.isOpen) {
      await this.client.quit();
      this.isConnected = false;
      console.log('✅ Redis disconnected gracefully');
    }
  }

  /**
   * Health check
   */
  async ping() {
    if (!this.isConnected || !this.client?.isOpen) {
      return false;
    }

    try {
      const result = await this.client.ping();
      return result === 'PONG';
    } catch (err) {
      return false;
    }
  }
}

module.exports = new RedisCache();