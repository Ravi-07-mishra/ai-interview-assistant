// db/connection.js - Enhanced MongoDB connection with pooling and retry logic
const mongoose = require("mongoose");

class DatabaseConnection {
  constructor() {
    this.isConnected = false;
    this.retryCount = 0;
    this.maxRetries = 5;
    this.retryDelay = 5000;
  }

  async connect() {
    if (this.isConnected) {
      console.log("✅ Using existing MongoDB connection");
      return;
    }

    const uri = process.env.MONGO_URI || "mongodb://127.0.0.1:27017/interviewdb";
    
    const options = {
      // Connection pooling
      maxPoolSize: 10,
      minPoolSize: 2,
      
      // Timeout settings
      serverSelectionTimeoutMS: 5000,
      socketTimeoutMS: 45000,
      
      // Automatic index creation (disable in production)
      autoIndex: process.env.NODE_ENV !== "production",
      
      // Retry writes
      retryWrites: true,
      retryReads: true,
      
      // Write concern
      w: "majority",
      wtimeoutMS: 2500,
      
      // Read preference
      readPreference: "primaryPreferred",
      
      // Compression
      compressors: ["zlib"],
    };

    try {
      await mongoose.connect(uri, options);
      this.isConnected = true;
      this.retryCount = 0;
      
      console.log("✅ MongoDB Connected Successfully");
      console.log(`   📊 Pool: ${options.maxPoolSize} max, ${options.minPoolSize} min`);
      
      // Setup connection event handlers
      this.setupEventHandlers();
      
    } catch (error) {
      console.error(`❌ MongoDB connection failed (attempt ${this.retryCount + 1}/${this.maxRetries}):`, error.message);
      
      if (this.retryCount < this.maxRetries) {
        this.retryCount++;
        console.log(`⏳ Retrying in ${this.retryDelay / 1000}s...`);
        await new Promise(resolve => setTimeout(resolve, this.retryDelay));
        return this.connect();
      }
      
      throw new Error(`Failed to connect to MongoDB after ${this.maxRetries} attempts`);
    }
  }

  setupEventHandlers() {
    mongoose.connection.on("disconnected", () => {
      console.warn("⚠️ MongoDB disconnected. Attempting to reconnect...");
      this.isConnected = false;
      this.connect();
    });

    mongoose.connection.on("error", (err) => {
      console.error("❌ MongoDB error:", err.message);
      this.isConnected = false;
    });

    mongoose.connection.on("reconnected", () => {
      console.log("✅ MongoDB reconnected");
      this.isConnected = true;
    });
  }

  async disconnect() {
    if (!this.isConnected) return;
    
    await mongoose.disconnect();
    this.isConnected = false;
    console.log("✅ MongoDB disconnected gracefully");
  }

  async healthCheck() {
    try {
      await mongoose.connection.db.admin().ping();
      return { status: "healthy", connected: this.isConnected };
    } catch (error) {
      return { status: "unhealthy", error: error.message };
    }
  }
}

module.exports = new DatabaseConnection();