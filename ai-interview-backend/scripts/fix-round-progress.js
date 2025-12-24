require("dotenv").config();
const db = require("../db/connection");
const Session = require("../models/Session");

async function fixRoundProgress() {
  await db.connect();

  const result = await Session.updateMany(
    {
      $or: [
        { "metadata.round_progress.screening": { $type: "number" } },
        { "metadata.round_progress.technical": { $type: "number" } },
        { "metadata.round_progress.behavioral": { $type: "number" } },
        { "metadata.round_progress": { $exists: false } }
      ]
    },
    {
      $set: {
        "metadata.round_progress": {
          screening: { questions: 0, status: "not_started" },
          technical: { questions: 0, status: "not_started" },
          behavioral: { questions: 0, status: "not_started" }
        }
      }
    }
  );

  console.log(`✅ Fixed ${result.modifiedCount} sessions`);
  await db.disconnect();
}

fixRoundProgress().catch(console.error);
