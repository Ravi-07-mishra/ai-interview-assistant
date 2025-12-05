# ai-interview-ai-service/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import os
from typing import Optional

app = FastAPI(title="AI Interview Service")

class EvalRequest(BaseModel):
    question_id: Optional[str] = None
    question_text: str
    ideal_outline: Optional[str] = ""
    candidate_answer: str
    resume_context: Optional[dict] = {}

@app.get("/")
def root():
    return {"status":"ok","service":"ai-interview-ai-service"}

@app.post("/evaluate")
def evaluate(req: EvalRequest):
    # Temporary dummy evaluator (replace with LLM logic later)
    words = len(req.candidate_answer.split())
    llm_score = min(10, max(0, words / 8))  # rough proxy
    embedding_score = round(min(1.0, words / 50), 3)
    heuristics_score = 10 if words >= 20 else (5 if words >= 10 else 2)

    final = round(0.7 * llm_score + 0.25 * (embedding_score * 10) + 0.05 * heuristics_score, 2)

    return {
        "score": final,
        "normalized_score": int(round(final*10)),
        "breakdown": {
            "llm_score": round(llm_score,2),
            "embedding_similarity": embedding_score,
            "heuristics_score": heuristics_score
        },
        "subscores": {
            "correctness": round(min(10, llm_score),2),
            "completeness": round(min(10, llm_score*0.9),2),
            "clarity": round(min(10, llm_score*0.9),2)
        },
        "summary": "This is a placeholder evaluation. Replace with LLM prompt & embeddings.",
        "improvements": ["Be more specific.", "Include an example.", "Structure your answer."]
    }

@app.post("/parse_resume")
def parse_resume(payload: dict):
    # Dummy parser: return a simple parsed JSON
    text = payload.get("text", "") or ""
    # naive skill extraction
    skills = []
    for k in ["python","java","react","sql","docker","ml","aws","fastapi","node"]:
        if k in text.lower():
            skills.append(k)
    return {"parsed": {"skills": skills, "summary": text[:400]}}
