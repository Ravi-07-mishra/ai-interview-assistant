# ai-interview-ai-service/main.py
# Enhanced AI Interview Service with Strict Technical Assessment
# Features:
#  - Deep technical question generation from resume projects
#  - Advanced bluff detection and gray-area scoring
#  - Aggressive early termination for poor candidates
#  - Multi-dimensional scoring with confidence tracking

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from groq import Groq
from dotenv import load_dotenv
from deepface import DeepFace
import cv2
from fastapi.responses import JSONResponse
import numpy as np
import base64
import os, io, json, re, logging
from typing import Dict
import time
FACE_DB: Dict[str, Dict[str, Any]] = {}  # sessionId -> {embedding: [...], thumbnail_b64: str, created: ts}

import pdfplumber, docx2txt, requests

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set in environment")

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")  # Using most capable model
MAX_PROMPT_CHARS = int(os.getenv("MAX_PROMPT_CHARS", "14000"))
DEFAULT_TOKEN_BUDGET = int(os.getenv("DEFAULT_TOKEN_BUDGET", "5000"))

groq_client = Groq(api_key=GROQ_API_KEY)

app = FastAPI(title="Enhanced AI Interview Service")
STRICT_DISTANCE_THRESHOLD = 0.55
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-service")
print("⏳ Loading VGG-Face model... please wait...")
DeepFace.build_model("VGG-Face")
print("✅ Model loaded!")
# ==========================================
# CORE CONFIGURATION
# ==========================================

TERMINATION_RULES = {
    "instant_fail_threshold": 0.20,   # Catastrophic failure
    "consecutive_fail_count": 2,      # Two fails in a row = reject (remains valid for efficiency)
    "consecutive_fail_threshold": 0.45,
    "excellence_threshold": 0.85,
    "min_confidence_to_end": 0.85,    # NEW: Model must be 85% sure to stop naturally
    "max_questions_soft_limit": 12,   # Expanded significantly, just to prevent infinite loops
    "gray_zone_min": 0.40,
    "gray_zone_max": 0.75,
}

SCORING_DIMENSIONS = {
    "technical_accuracy": {
        "weight": 0.40,
        "description": "Correctness of technical facts, algorithms, and concepts"
    },
    "depth_of_understanding": {
        "weight": 0.30,
        "description": "Ability to explain 'why' and 'how', not just 'what'"
    },
    "practical_experience": {
        "weight": 0.20,
        "description": "Evidence of real implementation, debugging, trade-offs"
    },
    "communication_clarity": {
        "weight": 0.10,
        "description": "Ability to articulate complex ideas clearly"
    }
}

# ==========================================
# UTILITIES
# ==========================================

EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
PHONE_RE = re.compile(r"(\+?\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}|\d{10})")

def extract_text_from_pdf_bytes(b: bytes) -> str:
    text_parts = []
    try:
        with pdfplumber.open(io.BytesIO(b)) as pdf:
            for page in pdf.pages:
                try:
                    txt = page.extract_text() or ""
                    if txt:
                        text_parts.append(txt)
                except Exception:
                    continue
    except Exception:
        return ""
    return "\n".join(text_parts)
import base64
import numpy as np
import cv2
import logging
from typing import Optional
import binascii # Import this for specific error catching

logger = logging.getLogger("ai-service")

def decode_base64_image(base64_string: str) -> Optional[np.ndarray]:
    """Decodes a Base64 string (data:image/jpeg;base64,...) to an OpenCV NumPy array."""
    if not base64_string:
        logger.error("Received empty base64 string.")
        return None

    original_start = base64_string[:50]
    
    # 1. Strip common prefixes (e.g., 'data:image/jpeg;base64,')
    if "," in base64_string:
        base64_string = base64_string.split(",", 1)[1]
    
    # Add a check for padding issues (DeepFace images are usually padded with '=')
    # base64.b64decode requires the input length to be a multiple of 4.
    padding_needed = len(base64_string) % 4
    if padding_needed != 0:
        base64_string += "=" * (4 - padding_needed)

    # Log the state after stripping/padding
    logger.info(f"[B64] Original Start: {original_start} | Final Length: {len(base64_string)}")

    try:
        # 2. Decode base64 to bytes (Note: DeepFace doesn't always send canonical Base64)
        image_bytes = base64.b64decode(base64_string, validate=True) 
        
        # 3. Convert bytes to a NumPy array
        np_arr = np.frombuffer(image_bytes, np.uint8)
        
        # 4. Decode the NumPy array into an image (OpenCV format)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            logger.error("cv2.imdecode failed, possibly invalid image data.")
        return img
    except binascii.Error as e:
        logger.error(f"[B64 ERROR] Base64 decoding failed: {e}. Check padding/chars.")
        return None
    except Exception as e:
        logger.error(f"[CONV ERROR] Image conversion failed: {e}")
        return None
def extract_text_from_docx_bytes(b: bytes) -> str:
    try:
        return docx2txt.process(io.BytesIO(b))
    except Exception:
        return ""
def make_thumbnail_b64(img: np.ndarray, max_w: int = 160) -> str:
    try:
        h, w = img.shape[:2]
        if w > max_w:
            scale = max_w / float(w)
            img_small = cv2.resize(img, (int(w * scale), int(h * scale)))
        else:
            img_small = img.copy()
        # convert BGR -> JPEG bytes
        ok, buf = cv2.imencode(".jpg", img_small, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        if not ok:
            return ""
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:
        return ""
def redact_pii(text: str) -> Dict[str, Any]:
    if not text:
        return {"redacted": "", "redaction_log": []}
    log = []
    for m in set(EMAIL_RE.findall(text)):
        log.append({"type":"email","value":m})
        text = text.replace(m, "[REDACTED_EMAIL]")
    text = PHONE_RE.sub("[REDACTED_PHONE]", text)
    return {"redacted": text, "redaction_log": log}

def extract_json_from_text(s: str) -> Optional[dict]:
    if not s:
        return None
    s = s.strip()
    
    # Try direct parse first
    try:
        return json.loads(s)
    except:
        pass
    
    # Remove markdown code fences
    s = re.sub(r'```json\s*', '', s)
    s = re.sub(r'```\s*', '', s)
    
    # Find JSON object
    start = s.find("{")
    if start == -1:
        return None
    
    stack = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            stack += 1
        elif s[i] == "}":
            stack -= 1
            if stack == 0:
                try:
                    return json.loads(s[start:i+1])
                except:
                    break
    return None

def safe_truncate(s: str, max_chars: int) -> str:
    if not s or len(s) <= max_chars:
        return s or ""
    return s[:max_chars-3] + "..."

def enforce_budget(payload: dict) -> dict:
    """Intelligent context truncation while preserving key information"""
    token_budget = int(payload.get("token_budget") or DEFAULT_TOKEN_BUDGET)
    char_limit = token_budget * 4
    
    resume = safe_truncate(payload.get("resume_summary",""), 1200)
    chunks = payload.get("retrieved_chunks",[])[:8]
    chunks = [
        {
            "doc_id": c.get("doc_id"),
            "chunk_id": c.get("chunk_id"),
            "snippet": safe_truncate(c.get("snippet",""), 600),
            "score": c.get("score", 0)
        }
        for c in chunks
    ]
    
    conv = payload.get("conversation", [])[-8:]
    conv = [{"role": t.get("role"), "text": safe_truncate(t.get("text",""), 500)} for t in conv]
    
    question = safe_truncate(payload.get("question",""), 1500)
    
    return {
        "resume": resume,
        "chunks": chunks,
        "conv": conv,
        "question": question
    }

def groq_call(prompt_text: str, model: Optional[str]=None, temperature: float=0.0, max_tokens: int=800) -> Dict[str, Any]:
    """Robust Groq API call with error handling"""
    model_name = model or GROQ_MODEL
    try:
        resp = groq_client.chat.completions.create(
            model=model_name,
            messages=[{"role":"user", "content": prompt_text}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        content = ""
        if hasattr(resp, "choices") and len(resp.choices) > 0:
            choice = resp.choices[0]
            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                content = choice.message.content or ""
            else:
                content = str(choice)
        else:
            content = str(resp)
            
        return {"raw": content, "ok": True}
    except Exception as e:
        logger.exception("groq_call failed")
        return {"raw": None, "ok": False, "error": str(e)}

# ==========================================
# PERFORMANCE ANALYTICS
# ==========================================

def calculate_performance_metrics(history: List[Dict]) -> Dict[str, Any]:
    """Calculate comprehensive performance metrics from question history"""
    if not history:
        return {
            "question_count": 0,
            "average_score": 0.0,
            "last_score": None,
            "consecutive_fails": 0,
            "consecutive_wins": 0,
            "trend": "unknown",
            "confidence": 0.0
        }
    
    scores = []
    for h in history:
        s = h.get("score")
        if s is not None:
            try:
                scores.append(float(s))
            except:
                pass
    
    if not scores:
        return {
            "question_count": len(history),
            "average_score": 0.0,
            "last_score": None,
            "consecutive_fails": 0,
            "consecutive_wins": 0,
            "trend": "unknown",
            "confidence": 0.0
        }
    
    avg_score = sum(scores) / len(scores)
    last_score = scores[-1]
    
    # Calculate streaks
    consecutive_fails = 0
    consecutive_wins = 0
    
    for s in reversed(scores):
        if s < TERMINATION_RULES["consecutive_fail_threshold"]:
            consecutive_fails += 1
            consecutive_wins = 0
        elif s > TERMINATION_RULES["excellence_threshold"]:
            consecutive_wins += 1
            consecutive_fails = 0
        else:
            break
    
    # Calculate trend
    if len(scores) >= 3:
        recent_avg = sum(scores[-3:]) / 3
        earlier_avg = sum(scores[:-3]) / len(scores[:-3]) if len(scores) > 3 else avg_score
        if recent_avg > earlier_avg + 0.15:
            trend = "improving"
        elif recent_avg < earlier_avg - 0.15:
            trend = "declining"
        else:
            trend = "stable"
    else:
        trend = "insufficient_data"
    
    # Calculate confidence based on consistency
    if len(scores) >= 3:
        variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
        std_dev = variance ** 0.5
        confidence = max(0.0, min(1.0, 1.0 - std_dev))
    else:
        confidence = 0.5
    
    return {
        "question_count": len(history),
        "average_score": avg_score,
        "last_score": last_score,
        "consecutive_fails": consecutive_fails,
        "consecutive_wins": consecutive_wins,
        "trend": trend,
        "confidence": confidence,
        "score_variance": variance if len(scores) >= 3 else 0.0
    }

# ==========================================
# TERMINATION LOGIC
# ==========================================

def check_termination_rules(history: List[Dict]) -> Optional[Dict[str, Any]]:
    """
    DYNAMIC TERMINATION LOGIC:
    - Ends immediately on Catastrophic Failure or Consecutive Fails (Efficiency).
    - Ends immediately on Proven Excellence (Saturation).
    - Otherwise, allows the LLM to decide when to stop until the Safety Limit is reached.
    """
    if not history:
        return None
    
    metrics = calculate_performance_metrics(history)
    rules = TERMINATION_RULES
    
    # RULE 1: Instant Fail - Catastrophic Answer
    # We keep this because a score < 0.2 means they don't know their own resume.
    if metrics["last_score"] is not None and metrics["last_score"] < rules["instant_fail_threshold"]:
        return {
            "ended": True,
            "verdict": "reject",
            "confidence": 0.98,
            "reason": f"Catastrophic answer detected (score: {metrics['last_score']:.2f}). Candidate lacks fundamental understanding.",
            "recommended_role": None,
            "trigger": "instant_fail"
        }
    
    # RULE 2: Consecutive Failures
    # We keep this for efficiency. 2 fails in a row usually guarantees a rejection.
    if metrics["consecutive_fails"] >= rules["consecutive_fail_count"]:
        return {
            "ended": True,
            "verdict": "reject",
            "confidence": 0.95,
            "reason": f"Failed {metrics['consecutive_fails']} consecutive technical questions. Technical competence not established.",
            "recommended_role": None,
            "trigger": "consecutive_fails"
        }
    
    # RULE 3: Consistent Excellence (Saturation)
    # If they ace 3 hard questions, we stop early to save time (Hire).
    if (metrics["question_count"] >= rules["excellence_count"] and 
        metrics["average_score"] > rules["excellence_threshold"] and
        metrics["consecutive_wins"] >= 2):
        return {
            "ended": True,
            "verdict": "hire",
            "confidence": 0.95,
            "reason": f"Candidate demonstrated consistent expert-level knowledge over {metrics['question_count']} questions.",
            "recommended_role": "Senior Engineer",
            "trigger": "excellence"
        }
    
    # RULE 4: Safety Net (Soft Limit)
    # Replaces the old "Max Questions" rule. This is just to prevent infinite loops.
    # The Model is expected to stop naturally BEFORE this point.
    soft_limit = rules.get("max_questions_soft_limit", 12)
    if metrics["question_count"] >= soft_limit:
        # If we reach here, the model was indecisive for too long. Force a decision.
        verdict = "hire" if metrics["average_score"] >= 0.65 else "reject"
        return {
            "ended": True,
            "verdict": verdict,
            "confidence": 0.80,
            "reason": f"Interview reached safety limit of {soft_limit} questions. Final decision forced based on average.",
            "recommended_role": "Mid-Level Engineer" if verdict == "hire" else None,
            "trigger": "safety_limit"
        }
    
    # If no hard rules are met, return None.
    # This passes control to 'call_decision', allowing the AI to decide 
    # if it needs more information or if it's ready to stop.
    return None

# ==========================================
# QUESTION GENERATION
# ==========================================

def extract_technical_projects(resume: str) -> List[Dict[str, str]]:
    """Extract technical projects from resume for targeted questioning"""
    projects = []
    
    # Look for project sections
    lines = resume.split('\n')
    current_project = None
    
    for i, line in enumerate(lines):
        line_lower = line.lower().strip()
        
        # Detect project headers
        if any(keyword in line_lower for keyword in ['project', 'built', 'developed', 'created', 'implemented']):
            if current_project:
                projects.append(current_project)
            
            current_project = {
                "title": line.strip()[:100],
                "description": "",
                "technologies": []
            }
        
        # Collect project details
        elif current_project:
            current_project["description"] += " " + line.strip()
            
            # Extract technologies
            tech_keywords = ['python', 'java', 'javascript', 'react', 'node', 'sql', 'aws', 
                           'docker', 'kubernetes', 'tensorflow', 'pytorch', 'api', 'rest', 'graphql']
            for tech in tech_keywords:
                if tech in line_lower and tech not in current_project["technologies"]:
                    current_project["technologies"].append(tech)
    
    if current_project:
        projects.append(current_project)
    
    return projects[:5]  # Return top 5 projects

def build_generate_question_prompt(context: dict, mode: str = "first") -> str:
    """
    Generate DEEP TECHNICAL questions that probe actual implementation knowledge.
    Uses 'Contextual Continuity' to ensure the interview flows logically based on performance.
    """
    resume = context.get("resume", "")
    chunks = context.get("chunks", [])
    conv = context.get("conv", [])
    history = context.get("history", [])
    
    # Extract projects for targeted questioning
    projects = extract_technical_projects(resume)
    
    # Build context
    chunks_text = "\n".join([
        f"[Source {c['doc_id']}:{c['chunk_id']}] {c.get('snippet','')[:400]}"
        for c in chunks[:4]
    ])
    
    conv_text = "\n".join([
        f"{m['role']}: {m['text'][:250]}"
        for m in conv[-4:]
    ])
    
    # Analyze what's been asked to avoid repetition
    asked_topics = set()
    if history:
        for h in history:
            q = h.get("question", "").lower()
            for tech in ['python', 'java', 'sql', 'api', 'algorithm', 'data structure', 'system design', 'aws', 'docker']:
                if tech in q:
                    asked_topics.add(tech)

    projects_text = ""
    if projects:
        projects_text = "IDENTIFIED TECHNICAL PROJECTS:\n"
        for i, p in enumerate(projects[:3], 1):
            projects_text += f"{i}. {p['title']}\n   Technologies: {', '.join(p['technologies']) or 'Not specified'}\n"

    # ==========================================
    # LOGIC: EXTRACT LAST INTERACTION DETAILS
    # ==========================================
    last_interaction_context = ""
    prev_score = 0.0
    
    if history:
        last_entry = history[-1]
        try:
            prev_score = float(last_entry.get("score", 0.0))
        except:
            prev_score = 0.0
            
        last_interaction_context = f"""
LAST INTERACTION DATA:
- Previous Question: "{last_entry.get('question', 'N/A')}"
- Candidate Answer: "{last_entry.get('answer', 'N/A')[:500]}"
- Score Received: {prev_score}
"""

    # ==========================================
    # MODE 1: FIRST QUESTION
    # ==========================================
    if mode == "first" or not history:
        system = """You are a HARDCORE Technical Interviewer at a top tech company. Your mission: Expose resume fluff.

BANNED QUESTIONS (never ask these):
- "Tell me about yourself"
- "What are your strengths/weaknesses"
- "Why do you want this role"
- Any generic behavioral question

REQUIRED QUESTION TYPES:
1. **Implementation Deep-Dive**: "In [specific project], how exactly did you implement [specific feature]? Walk me through the core logic."
2. **Trade-off Analysis**: "Why did you choose [technology X] over [alternative Y]? What were the performance implications?"
3. **Debugging Scenario**: "What was the hardest bug you encountered in [project]? How did you diagnose it?"

Your questions MUST:
- Reference a SPECIFIC project/technology from their resume
- Require implementation-level knowledge to answer
- Be impossible to answer with generic preparation"""

        schema = '''{
  "question": "string (detailed, project-specific question)",
  "target_project": "string (which resume project this targets)",
  "technology_focus": "string (specific tech being tested)",
  "expected_answer_type": "medium|code|architectural",
  "difficulty": "hard|expert",
  "ideal_answer_outline": "string (what a strong answer covers)",
  "red_flags": ["list of signs the candidate is bluffing"],
  "confidence": 0.0-1.0
}'''

        prompt = f"""SYSTEM: {system}

CANDIDATE RESUME:
```
{resume[:1500]}
```

{projects_text}

CONTEXT DOCUMENTS:
{chunks_text}

INSTRUCTION:
1. Pick the MOST IMPRESSIVE or COMPLEX project from the resume.
2. Identify a specific technical claim (algorithm, architecture, optimization).
3. Ask a question that ONLY someone who actually implemented it could answer.
4. Output strict JSON:
{schema}
"""

    # ==========================================
    # MODE 2: FOLLOW-UP (CONTEXTUAL CONTINUITY)
    # ==========================================
    else:
        system = """You are a Dynamic Technical Interviewer. You adapt your questioning based on the candidate's previous performance.

RULES FOR NEXT QUESTION (Contextual Continuity):
1. **IF PREV SCORE < 0.5 (REMEDIAL)**:
   - The candidate failed the previous question. 
   - Ask a SIMPLER, fundamental question on the SAME topic.
   - Goal: Check if they lack deep knowledge or just misunderstood the complex question.
   
2. **IF PREV SCORE 0.5 - 0.79 (DEEP DIVE)**:
   - The candidate gave a surface-level or partial answer.
   - DRILL DEEPER into the exact same topic.
   - Ask for code-level specifics, specific libraries used, or how they handled a specific edge case mentioned in their answer.
   
3. **IF PREV SCORE >= 0.8 (NEW CHALLENGE)**:
   - The candidate mastered the last topic.
   - SWITCH TOPICS completely. Pick a different project or skill from the resume.
   - Do NOT ask about: {asked_topics_list}

GENERAL RULES:
- Never say "Great answer". Just ask the next question.
- Keep questions technical and implementation-focused.
"""

        schema = '''{
  "question": "string",
  "strategy_used": "remedial|deep_dive|new_topic",
  "target_project": "string",
  "technology_focus": "string",
  "difficulty": "medium|hard|expert",
  "ideal_answer_outline": "string",
  "red_flags": ["list"],
  "confidence": 0.0-1.0
}'''

        prompt = f"""SYSTEM: {system}

RESUME PROJECTS:
{projects_text}

ALREADY DISCUSSED TOPICS: {list(asked_topics)}

{last_interaction_context}

INSTRUCTION:
Based on the Previous Score ({prev_score}), generate the next logical question using the strategy rules above.
Output JSON: {schema}
"""
    
    return prompt.strip()

# ==========================================
# SCORING SYSTEM
# ==========================================

def build_score_prompt(question_text: str, ideal_outline: str, candidate_answer: str, context: dict = None) -> str:
    """
    STRICT MULTI-DIMENSIONAL SCORING with advanced bluff detection.
    """
    context = context or {}
    resume = context.get("resume", "")
    chunks = context.get("chunks", [])
    
    chunks_text = ""
    if chunks:
        chunks_text = "\n".join([
            f"[Ref {c.get('doc_id')}:{c.get('chunk_id')}] {c.get('snippet','')[:300]}"
            for c in chunks[:3]
        ])
    
    # Assuming SCORING_DIMENSIONS is defined globally as per your previous code context
    dimensions_text = "\n".join([
        f"- **{name}** (weight: {info['weight']}): {info['description']}"
        for name, info in SCORING_DIMENSIONS.items()
    ])
    
    system = f"""You are an EXPERT Technical Assessor specializing in detecting resume fraud and technical incompetence.

SCORING DIMENSIONS:
{dimensions_text}

SCORING RULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**FAIL (0.0 - 0.3)**: One or more of:
  • Answer is completely off-topic or nonsensical
  • Uses buzzwords incorrectly (e.g., "used machine learning" but can't explain gradient descent)
  • Admits "I don't remember" or "I wasn't the main person" for their OWN resume project
  • Gives textbook definition when asked for implementation details
  • Contradicts basic computer science principles

**WEAK (0.3 - 0.5)**: 
  • Generic answer that could come from a blog post
  • Mentions correct concepts but can't explain "why" or "how"
  • Avoids the specific question and talks about something easier
  • No evidence of hands-on implementation

**ACCEPTABLE (0.5 - 0.7)**:
  • Demonstrates basic understanding
  • Mentions specific technologies correctly
  • Some implementation details but missing depth
  • Could have learned this from a tutorial

**STRONG (0.7 - 0.85)**:
  • Specific implementation details (file structures, class names, algorithms)
  • Explains trade-offs made during development
  • Discusses challenges encountered and how they solved them
  • Clear evidence they actually built it

**EXCEPTIONAL (0.85 - 1.0)**:
  • Discusses performance metrics, edge cases, production issues
  • Compares multiple approaches with specific pros/cons
  • Shows deep understanding of underlying principles
  • Could teach others how to implement it
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CRITICAL BLUFF DETECTORS:
1. **Vague Language**: "we used industry best practices", "implemented modern solutions"
2. **Passive Voice**: "it was done", "the system was built" (who built it?)
3. **Deflection**: Answering a different, easier question
4. **Buzzword Salad**: Using terms without connecting them logically
5. **No Specifics**: Can't name files, functions, algorithms, or metrics"""

    # UPDATED SCHEMA: Added mentor_tip
    schema = '''{
  "overall_score": 0.0-1.0,
  "dimension_scores": {
    "technical_accuracy": 0.0-1.0,
    "depth_of_understanding": 0.0-1.0,
    "practical_experience": 0.0-1.0,
    "communication_clarity": 0.0-1.0
  },
  "confidence": 0.0-1.0,
  "verdict": "fail|weak|acceptable|strong|exceptional",
  "rationale": "string (specific evidence for score)",
  "red_flags_detected": ["list of bluff indicators found"],
  "missing_elements": ["what a strong answer would have included"],
  "follow_up_probe": "string (question to clarify gray areas)" or null,
  "mentor_tip": "string (constructive advice or specific learning resource based on the gaps in this specific answer)"
}'''

    prompt = f"""SYSTEM: {system}

QUESTION ASKED:
```
{question_text}
```

IDEAL ANSWER SHOULD COVER:
```
{ideal_outline}
```

CANDIDATE'S ANSWER:
```
{candidate_answer[:800]}
```

RESUME CONTEXT (for verification):
```
{resume[:600]}
```

REFERENCE MATERIALS:
{chunks_text}

INSTRUCTION:
1. Evaluate each dimension independently
2. Calculate weighted overall score
3. Identify specific red flags (vague language, deflection, incorrect usage)
4. Be HARSH on generic answers - most candidates exaggerate
5. If answer is in gray zone (0.4-0.7), suggest a follow-up probe
6. Provide a specific MENTOR TIP: Address the specific gap in their logic or depth. What should they read or practice to improve this specific answer?

Output JSON: {schema}
"""
    
    return prompt.strip()

# ==========================================
# DECISION ENGINE
# ==========================================

def build_decision_prompt(context: dict) -> str:
    """Generate comprehensive decision prompt with performance analytics"""
    resume = context.get("resume", "")
    history = context.get("question_history", [])
    
    metrics = calculate_performance_metrics(history)
    
    # Build question history summary
    history_text = ""
    for i, h in enumerate(history[-6:], 1):
        q = h.get("question", "")[:150]
        a = h.get("answer", "")[:200]
        score = h.get("score")
        verdict = h.get("verdict", "N/A")
        
        history_text += f"""
Question {i}: {q}
Answer: {a}
Score: {score} ({verdict})
---"""
    
    termination_guidance = ""
    if metrics["consecutive_fails"] >= 2:
        termination_guidance = "⚠️ CRITICAL: 2+ consecutive fails detected. You MUST reject unless there's exceptional justification."
    elif metrics["average_score"] < 0.50 and metrics["question_count"] >= 3:
        termination_guidance = "⚠️ CRITICAL: Average below hiring bar. Recommend reject."
    elif metrics["consecutive_wins"] >= 3:
        termination_guidance = "✓ STRONG SIGNAL: Consistent excellence detected. Consider hiring."
    elif metrics["question_count"] >= 7:
        termination_guidance = "⏰ TIME LIMIT: Must make final decision now."
    
    schema = '''{
  "ended": boolean,
  "verdict": "hire|reject|maybe",
  "confidence": 0.0-1.0,
  "reason": "string (specific, evidence-based)",
  "recommended_role": "string|null",
  "key_strengths": ["list"],
  "critical_weaknesses": ["list"]
}'''

    prompt = f"""You are a Senior Hiring Manager.
    
    METRICS:
    Questions: {metrics['question_count']}
    Avg Score: {metrics['average_score']:.2f}
    Confidence: {metrics['confidence']:.2f}
    
    INTERVIEW HISTORY:
    {history_text}
    
    DECISION LOGIC:
    1. **CONTINUE (ended: false)**: If you are unsure (Confidence < {TERMINATION_RULES['min_confidence_to_end']}) and need to probe more skills.
    2. **HIRE (ended: true)**: If candidate showed clear expertise across multiple topics.
    3. **REJECT (ended: true)**: If candidate failed basic questions or was caught bluffing.
    
    **CRITICAL**: Do not stop early just to be short. Only stop if you have CONCRETE evidence for a Hire/Reject decision.
    
    Output JSON: {{
      "ended": boolean,
      "verdict": "hire|reject|maybe",
      "confidence": 0.0-1.0,
      "reason": "string"
    }}
    """
    return prompt.strip()

def call_decision(context: dict, temperature: float = 0.0) -> Dict[str, Any]:
    """Make hiring decision with performance-based termination"""
    # First check hard rules
    hard_decision = check_termination_rules(context.get("question_history", []))
    if hard_decision:
        return {"ok": True, "parsed": hard_decision, "raw": "hard_rule_triggered"}
    
    # Otherwise, consult the model
    prompt = build_decision_prompt(context)
    resp = groq_call(prompt, temperature=temperature, max_tokens=600)
    
    if not resp.get("ok"):
        return {"ok": False, "parsed": None, "raw": resp.get("error")}
    
    parsed = extract_json_from_text(resp["raw"])
    if not parsed:
        # Fallback decision based on metrics
        metrics = calculate_performance_metrics(context.get("question_history", []))
        return {
            "ok": True,
            "parsed": {
                "ended": metrics["question_count"] >= 6,
                "verdict": "maybe",
                "confidence": 0.5,
                "reason": "Model parse failed, using metrics fallback",
                "recommended_role": None
            },
            "raw": resp["raw"]
        }
    
    # Normalize the parsed decision
    normalized = {
        "ended": bool(parsed.get("ended", False)),
        "verdict": parsed.get("verdict", "maybe"),
        "confidence": float(parsed.get("confidence", 0.5)),
        "reason": parsed.get("reason", ""),
        "recommended_role": parsed.get("recommended_role"),
        "key_strengths": parsed.get("key_strengths", []),
        "critical_weaknesses": parsed.get("critical_weaknesses", [])
    }
    
    # Force termination if max questions reached
    if context.get("question_history") and len(context["question_history"]) >= TERMINATION_RULES["max_questions"]:
        normalized["ended"] = True
    
    return {"ok": True, "parsed": normalized, "raw": resp["raw"]}

# ==========================================
# PROBE GENERATION
# ==========================================

def build_probe_prompt(weakness_topic: str, prev_question: str, prev_answer: str, context: dict) -> str:
    """Generate targeted probe questions for gray-zone answers"""
    resume = context.get("resume", "")
    
    system = """You are a Technical Interviewer conducting a diagnostic probe.

PROBE STRATEGY:
The candidate gave a VAGUE or INCOMPLETE answer. Your job: Get them to be SPECIFIC.

Good probes:
- "Can you show me the actual code structure for that?"
- "What specific error did you encounter, and what was in the stack trace?"
- "Walk me through the exact steps your algorithm takes with a sample input."
- "Why did you choose approach X over Y? What were the performance numbers?"

Bad probes:
- "Can you tell me more?" (too open-ended)
- "That's interesting, continue" (not diagnostic)
"""

    schema = '''{
  "probe_question": "string (specific, forces concrete details)",
  "what_to_listen_for": ["specific signals of real knowledge"],
  "red_flags_if_missing": ["signs they're still bluffing"],
  "difficulty": "medium|hard",
  "expected_answer_length": "short|medium",
  "scoring_criteria": ["list of what makes answer acceptable"]
}'''

    prompt = f"""SYSTEM: {system}

WEAKNESS DETECTED: {weakness_topic}

ORIGINAL QUESTION:
```
{prev_question[:300]}
```

CANDIDATE'S VAGUE ANSWER:
```
{prev_answer[:400]}
```

RESUME CONTEXT:
```
{resume[:500]}
```

INSTRUCTION:
Generate a probe that forces the candidate to provide:
1. Specific technical details (code, algorithms, architectures)
2. Concrete examples (actual bug, real metric, specific file)
3. Evidence they personally implemented it (not just "we" or "the team")

Output JSON: {schema}
"""
    
    return prompt.strip()

def call_probe(weakness_topic: str, prev_question: str, prev_answer: str, context: dict) -> Dict[str, Any]:
    """Generate probe with fallback"""
    prompt = build_probe_prompt(weakness_topic, prev_question, prev_answer, context)
    resp = groq_call(prompt, temperature=0.0, max_tokens=500)
    
    if not resp.get("ok"):
        # Fallback probe
        return {
            "ok": True,
            "parsed": {
                "probe_question": f"Can you provide a specific code example or pseudocode for how you implemented {weakness_topic}?",
                "what_to_listen_for": ["code structure", "algorithm steps", "specific libraries"],
                "red_flags_if_missing": ["still vague", "changes subject", "uses 'we' without 'I'"],
                "difficulty": "medium",
                "expected_answer_length": "medium",
                "scoring_criteria": ["provides code or detailed logic", "explains specific choices"]
            },
            "raw": "fallback"
        }
    
    parsed = extract_json_from_text(resp["raw"])
    if not parsed:
        return call_probe(weakness_topic, prev_question, prev_answer, context)  # Retry once
    
    return {"ok": True, "parsed": parsed, "raw": resp["raw"]}

# ==========================================
# RESUME PARSING
# ==========================================

def groq_parse_resume(text: str) -> dict:
    """Enhanced resume parser with AI + regex fallback"""
    prompt = f"""Extract structured information from this resume. Return ONLY valid JSON:

{{
  "name": "string|null",
  "email": "string|null", 
  "phone": "string|null",
  "skills": ["list of technical skills"],
  "experience_years": number|null,
  "education": [{{"degree": "string", "institution": "string", "year": "string|null"}}],
  "projects": [{{"title": "string", "technologies": ["list"], "description": "string"}}],
  "work_experience": [{{"company": "string", "role": "string", "duration": "string", "responsibilities": ["list"]}}],
  "summary": "string (2-3 sentence overview)"
}}

Resume:
```
{text[:4000]}
```"""

    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1500
        )
        
        content = resp.choices[0].message.content if resp.choices else str(resp)
        parsed = extract_json_from_text(content)
        
        if parsed and (parsed.get("name") or parsed.get("email") or parsed.get("skills")):
            return parsed
    except Exception as e:
        logger.warning(f"AI resume parsing failed: {e}")
    
    # Fallback to regex extraction
    logger.info("Using regex fallback for resume parsing")
    return regex_parse_resume(text)

def regex_parse_resume(text: str) -> dict:
    """Robust regex-based resume parser"""
    # Extract name (first non-header line with 2-4 capitalized words)
    name = None
    for line in text.split('\n')[:10]:
        line = line.strip()
        if line and not any(kw in line.lower() for kw in ['resume', 'cv', 'curriculum', 'profile', '@']):
            words = line.split()
            if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w):
                name = line
                break
    
    # Extract contact info
    email = EMAIL_RE.search(text)
    phone = PHONE_RE.search(text)
    
    # Extract skills
    common_skills = [
        'Python', 'Java', 'JavaScript', 'TypeScript', 'C++', 'C#', 'Go', 'Rust', 'Ruby',
        'React', 'Angular', 'Vue', 'Node.js', 'Django', 'Flask', 'Spring', 'Express',
        'SQL', 'PostgreSQL', 'MySQL', 'MongoDB', 'Redis', 'Elasticsearch',
        'AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes', 'Jenkins', 'GitLab', 'GitHub Actions',
        'TensorFlow', 'PyTorch', 'Scikit-learn', 'Pandas', 'NumPy',
        'REST API', 'GraphQL', 'Microservices', 'Agile', 'Scrum'
    ]
    
    skills = []
    text_lower = text.lower()
    for skill in common_skills:
        if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text_lower):
            skills.append(skill)
    
    # Extract education
    education = []
    edu_keywords = ['bachelor', 'master', 'phd', 'b.tech', 'm.tech', 'b.sc', 'm.sc', 'diploma', 'degree']
    for line in text.split('\n'):
        if any(kw in line.lower() for kw in edu_keywords):
            education.append({
                "degree": line.strip()[:100],
                "institution": "Not specified",
                "year": None
            })
    
    # Extract projects
    projects = []
    lines = text.split('\n')
    for i, line in enumerate(lines):
        line_lower = line.lower()
        if any(kw in line_lower for kw in ['project:', 'built', 'developed', 'created', 'implemented']):
            description = ' '.join(lines[i:i+3])[:200]
            proj_skills = [s for s in skills if s.lower() in description.lower()]
            projects.append({
                "title": line.strip()[:100],
                "technologies": proj_skills[:5],
                "description": description
            })
    
    return {
        "name": name or "Candidate",
        "email": email.group(0) if email else None,
        "phone": phone.group(0) if phone else None,
        "skills": skills[:15],
        "experience_years": None,
        "education": education[:3],
        "projects": projects[:5],
        "work_experience": [],
        "summary": safe_truncate(text.replace('\n', ' ').strip(), 300)
    }

# ==========================================
# API MODELS
# ==========================================

class GenerateQuestionRequest(BaseModel):
    request_id: str
    session_id: str
    user_id: str
    mode: Optional[str] = "first"
    resume_summary: Optional[str] = ""
    retrieved_chunks: Optional[List[Dict[str,Any]]] = []
    conversation: Optional[List[Dict[str,str]]] = []
    question_history: Optional[List[Dict[str,Any]]] = []
    token_budget: Optional[int] = DEFAULT_TOKEN_BUDGET
    allow_pii: Optional[bool] = False
    options: Optional[Dict[str,Any]] = {}
class FaceVerificationRequest(BaseModel):
    reference_image: str
    current_image: str
class FaceRegisterRequest(BaseModel):
    sessionId: str
    image: str  # This is the base64 string    
class ScoreAnswerRequest(BaseModel):
    request_id: str
    session_id: str
    user_id: str
    question_text: str
    ideal_outline: str
    candidate_answer: str
    resume_summary: Optional[str] = ""
    retrieved_chunks: Optional[List[Dict[str,Any]]] = []
    question_history: Optional[List[Dict[str,Any]]] = []
    token_budget: Optional[int] = DEFAULT_TOKEN_BUDGET
    allow_pii: Optional[bool] = False
    options: Optional[Dict[str,Any]] = {}

class ProbeRequest(BaseModel):
    request_id: str
    session_id: str
    user_id: str
    weakness_topic: str
    prev_question: str
    prev_answer: str
    resume_summary: Optional[str] = ""
    retrieved_chunks: Optional[List[Dict[str,Any]]] = []
    conversation: Optional[List[Dict[str,str]]] = []
    token_budget: Optional[int] = DEFAULT_TOKEN_BUDGET
    allow_pii: Optional[bool] = False
    options: Optional[Dict[str,Any]] = {}

class DecisionRequest(BaseModel):
    request_id: str
    session_id: str
    user_id: str
    resume_summary: Optional[str] = ""
    conversation: Optional[List[Dict[str,str]]] = []
    question_history: List[Dict[str,Any]]
    retrieved_chunks: Optional[List[Dict[str,Any]]] = []
    token_budget: Optional[int] = DEFAULT_TOKEN_BUDGET
    allow_pii: Optional[bool] = False
    accept_model_final: Optional[bool] = True

# ==========================================
# API ENDPOINTS
# ==========================================

@app.post("/generate_question")
def generate_question(req: GenerateQuestionRequest):
    """Generate deep technical question targeting resume claims"""
    payload = req.dict()
    
    # PII redaction
    redaction_log = []
    if not payload.get("allow_pii") and payload.get("resume_summary"):
        r = redact_pii(payload["resume_summary"])
        payload["resume_summary"] = r["redacted"]
        redaction_log = r["redaction_log"]
    
    enforced = enforce_budget(payload)
    
    # Add history to context
    enforced["history"] = payload.get("question_history", [])
    
    prompt = build_generate_question_prompt(enforced, mode=payload.get("mode", "first"))
    
    options = payload.get("options", {})
    temperature = float(options.get("temperature", 0.1))  # Slight randomness for variety
    max_tokens = int(options.get("max_response_tokens", 800))
    
    if len(prompt) > MAX_PROMPT_CHARS:
        raise HTTPException(status_code=400, detail="prompt_too_large")
    
    resp = groq_call(prompt, temperature=temperature, max_tokens=max_tokens)
    
    if not resp.get("ok"):
        raise HTTPException(status_code=502, detail=f"groq_error: {resp.get('error')}")
    
    parsed = extract_json_from_text(resp["raw"])
    
    # Fallback question if parsing failed
    if not parsed or not isinstance(parsed, dict):
        resume = enforced.get("resume", "")
        skills = []
        for skill in ['Python', 'JavaScript', 'Java', 'SQL', 'React', 'Node.js', 'AWS', 'Docker']:
            if skill.lower() in resume.lower():
                skills.append(skill)
        
        target_skill = skills[0] if skills else "your most complex project"
        
        parsed = {
            "question": f"Tell me about the most technically challenging aspect of {target_skill} in your experience. What specific problem did you solve, and how did you approach it?",
            "target_project": "General",
            "technology_focus": target_skill,
            "expected_answer_type": "medium",
            "difficulty": "hard",
            "ideal_answer_outline": "Should describe: specific technical challenge, approach taken, implementation details, trade-offs considered, outcome achieved",
            "red_flags": ["vague descriptions", "no specific implementation details", "team did it without personal contribution"],
            "confidence": 0.4
        }
    
    return {
        "request_id": payload["request_id"],
        "prompt_text": prompt if options.get("return_prompt") else None,
        "llm_raw": resp["raw"],
        "parsed": parsed,
        "redaction_log": redaction_log
    }
@app.post("/interview/register-face")
async def register_face(request: FaceRegisterRequest):
    """
    Decode, validate, detect face, create embedding, and store it for the session.
    Returns 400 if image invalid / no face detected so frontend won't proceed.
    """
    try:
        if not request.image:
            raise HTTPException(status_code=400, detail="No image provided")

        # 1) decode base64 into OpenCV image
        img = decode_base64_image(request.image)
        if img is None:
            raise HTTPException(status_code=400, detail="Image decoding failed. Check Base64/data URL format.")

        # 2) quick quality checks (brightness + contrast/variance)
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mean, stddev = cv2.meanStdDev(gray)
            mean_val = float(mean[0][0])
            stddev_val = float(stddev[0][0])
        except Exception as e:
            logger.warning("Image quality check failed to compute stats: %s", e)
            mean_val, stddev_val = 0.0, 0.0

        if mean_val < 16:
            raise HTTPException(status_code=400, detail=f"Captured image too dark (mean={mean_val:.1f}). Improve lighting.")
        if mean_val > 250:
            raise HTTPException(status_code=400, detail=f"Captured image too bright (mean={mean_val:.1f}). Avoid bright backlight.")
        if stddev_val < 6:
            raise HTTPException(status_code=400, detail=f"Captured image low-contrast or blurry (stddev={stddev_val:.1f}). Please hold still and ensure face is focused.")

        # 3) attempt to extract an embedding via DeepFace (enforce_detection => raises if no face)
        try:
            # DeepFace.represent returns a list of embeddings when given an image array
            # We force enforce_detection=True so it raises if a face isn't found.
            rep = DeepFace.represent(img_path=img, model_name="VGG-Face", detector_backend="mtcnn", enforce_detection=True)
            # The representation can be returned in different formats depending on deepface version.
            # Normalize to a plain list of floats
            embedding = None
            if isinstance(rep, list) and len(rep) > 0:
                # rep might be a list of dicts or list of vectors
                first = rep[0]
                if isinstance(first, dict) and "embedding" in first:
                    embedding = list(map(float, first["embedding"]))
                elif isinstance(first, (list, tuple, np.ndarray)):
                    embedding = [float(x) for x in first]
                else:
                    # best-effort fallback
                    embedding = [float(x) for x in np.array(first).reshape(-1).tolist()]
            elif isinstance(rep, (np.ndarray, list, tuple)):
                # fallback convert
                embedding = [float(x) for x in np.array(rep).reshape(-1).tolist()]

            if not embedding or len(embedding) < 50:
                # embedding length check - VGG-Face embeddings are large (~2622 in some builds) but we just sanity-check
                logger.warning("Unexpected embedding form/length from DeepFace: len=%s", None if embedding is None else len(embedding))
                raise HTTPException(status_code=500, detail="Failed to extract face embedding (unexpected format).")
        except ValueError as e:
            # Typical DeepFace message when no face detected
            logger.warning("DeepFace enforce_detection error during register_face: %s", e)
            raise HTTPException(status_code=400, detail="No face detected in reference image. Please align your face and try again.")
        except Exception as e:
            logger.exception("Unexpected DeepFace error during register_face")
            raise HTTPException(status_code=500, detail=f"Face processing failed: {str(e)}")

        # 4) store embedding + small diagnostic thumbnail in memory (persist to DB in prod)
        try:
            thumb_b64 = make_thumbnail_b64(img)
            FACE_DB[request.sessionId] = {
                "embedding": embedding,
                "thumbnail": thumb_b64,
                "created": time.time(),
                "mean_brightness": mean_val,
                "stddev": stddev_val
            }
            logger.info("Registered face for session=%s; embedding_len=%d mean=%.1f std=%.1f", request.sessionId, len(embedding), mean_val, stddev_val)
        except Exception as e:
            logger.exception("Failed to store face registration")
            raise HTTPException(status_code=500, detail="Server failed to store face registration.")

        # 5) Respond success (frontend should require resp.ok before proceeding)
        return {"status": "registered", "message": "Face identity saved", "sessionId": request.sessionId}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error in register_face")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
@app.post("/score_answer")
def score_answer(req: ScoreAnswerRequest):
    """Score answer with multi-dimensional evaluation and bluff detection"""
    payload = req.dict()
    
    # PII redaction
    redaction_log = []
    if not payload.get("allow_pii") and payload.get("resume_summary"):
        r = redact_pii(payload["resume_summary"])
        payload["resume_summary"] = r["redacted"]
        redaction_log = r["redaction_log"]
    
    enforced = enforce_budget(payload)
    
    context = {
        "resume": enforced.get("resume", ""),
        "chunks": enforced.get("chunks", [])
    }
    
    prompt = build_score_prompt(
        payload.get("question_text", ""),
        payload.get("ideal_outline", ""),
        payload.get("candidate_answer", ""),
        context
    )
    
    options = payload.get("options", {})
    temperature = float(options.get("temperature", 0.0))
    max_tokens = int(options.get("max_response_tokens", 1000))
    
    if len(prompt) > MAX_PROMPT_CHARS:
        raise HTTPException(status_code=400, detail="prompt_too_large")
    
    resp = groq_call(prompt, temperature=temperature, max_tokens=max_tokens)
    
    if not resp.get("ok"):
        raise HTTPException(status_code=502, detail=f"groq_error: {resp.get('error')}")
    
    parsed = extract_json_from_text(resp["raw"])
    
    # Validate and normalize scores
    validated = {
        "overall_score": None,
        "dimension_scores": {},
        "confidence": 0.5,
        "verdict": "weak",
        "rationale": "",
        "red_flags_detected": [],
        "missing_elements": [],
        "follow_up_probe": None
    }
    
    needs_review = False
    
    if parsed and isinstance(parsed, dict):
        try:
            # Overall score
            score = parsed.get("overall_score")
            if score is not None:
                validated["overall_score"] = max(0.0, min(1.0, float(score)))
            
            # Dimension scores
            dim_scores = parsed.get("dimension_scores", {})
            for dim in SCORING_DIMENSIONS.keys():
                val = dim_scores.get(dim)
                if val is not None:
                    validated["dimension_scores"][dim] = max(0.0, min(1.0, float(val)))
            
            # Other fields
            validated["confidence"] = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
            validated["verdict"] = parsed.get("verdict", "weak")
            validated["rationale"] = parsed.get("rationale", "")
            validated["red_flags_detected"] = parsed.get("red_flags_detected", [])
            validated["missing_elements"] = parsed.get("missing_elements", [])
            validated["follow_up_probe"] = parsed.get("follow_up_probe")
            
            # Gray zone detection
            if validated["overall_score"] is not None:
                rules = TERMINATION_RULES
                if rules["gray_zone_min"] <= validated["overall_score"] <= rules["gray_zone_max"]:
                    needs_review = True
                    if not validated["follow_up_probe"]:
                        validated["follow_up_probe"] = "Ask for specific code example or implementation detail"
            
            # Low confidence flag
            if validated["confidence"] < 0.4:
                needs_review = True
            
        except Exception as e:
            logger.exception(f"Score validation failed: {e}")
            needs_review = True
    else:
        needs_review = True
    
    return {
        "request_id": payload["request_id"],
        "llm_raw": resp["raw"],
        "parsed": parsed,
        "validated": validated,
        "parse_ok": parsed is not None,
        "needs_human_review": needs_review,
        "in_gray_zone": needs_review and validated["overall_score"] is not None and 
                        TERMINATION_RULES["gray_zone_min"] <= validated["overall_score"] <= TERMINATION_RULES["gray_zone_max"],
        "redaction_log": redaction_log
    }

@app.post("/probe")
def probe(req: ProbeRequest):
    """Generate diagnostic probe question for weak/vague answers"""
    payload = req.dict()
    
    # PII redaction
    redaction_log = []
    if not payload.get("allow_pii") and payload.get("resume_summary"):
        r = redact_pii(payload["resume_summary"])
        payload["resume_summary"] = r["redacted"]
        redaction_log = r["redaction_log"]
    
    enforced = enforce_budget(payload)
    
    context = {
        "resume": enforced.get("resume", ""),
        "chunks": enforced.get("chunks", []),
        "conv": enforced.get("conv", [])
    }
    
    probe_result = call_probe(
        payload.get("weakness_topic", ""),
        payload.get("prev_question", ""),
        payload.get("prev_answer", ""),
        context
    )
    
    return {
        "request_id": payload["request_id"],
        "llm_raw": probe_result.get("raw"),
        "parsed": probe_result.get("parsed"),
        "redaction_log": redaction_log
    }

@app.post("/finalize_decision")
def finalize_decision(req: DecisionRequest):
    """Make final hiring decision with performance-based termination"""
    payload = req.dict()
    
    # PII redaction
    if not payload.get("allow_pii") and payload.get("resume_summary"):
        r = redact_pii(payload["resume_summary"])
        payload["resume_summary"] = r["redacted"]
    
    enforced = enforce_budget(payload)
    
    context = {
        "resume": enforced.get("resume", ""),
        "conversation": payload.get("conversation", []),
        "question_history": payload.get("question_history", []),
        "retrieved_chunks": enforced.get("chunks", [])
    }
    
    result = call_decision(context, temperature=0.0)
    
    # Determine if decision is final
    is_final = False
    if result.get("ok") and result.get("parsed"):
        decision = result["parsed"]
        if decision.get("ended"):
            verdict = decision.get("verdict")
            confidence = decision.get("confidence", 0.0)
            
            # Accept as final if confident and clear verdict
            if payload.get("accept_model_final", True):
                if verdict in ("hire", "reject") and confidence >= 0.75:
                    is_final = True
                elif verdict == "maybe" and confidence < 0.5:
                    is_final = False  # Needs human review
    
    # Add performance metrics
    metrics = calculate_performance_metrics(payload.get("question_history", []))
    
    return {
        "request_id": payload["request_id"],
        "result": result,
        "is_final": is_final,
        "performance_metrics": metrics,
        "termination_rule_triggered": result.get("raw") == "hard_rule_triggered"
    }

@app.post("/parse_resume")
async def parse_resume(
    file: UploadFile = File(None),
    s3_url: Optional[str] = Form(None),
    text: Optional[str] = Form(None),
    resume_id: Optional[str] = Form(None)
):
    """Parse resume into structured format with AI + fallback"""
    raw_text = ""
    
    # Extract text from various sources
    if file is not None:
        contents = await file.read()
        filename = (file.filename or "").lower()
        
        if filename.endswith(".pdf"):
            raw_text = extract_text_from_pdf_bytes(contents)
        elif filename.endswith(".docx"):
            raw_text = extract_text_from_docx_bytes(contents)
        else:
            try:
                raw_text = contents.decode("utf-8", errors="ignore")
            except:
                raw_text = ""
                
    elif text:
        raw_text = text
        
    elif s3_url:
        try:
            r = requests.get(s3_url, timeout=15)
            if r.status_code == 200:
                content_type = r.headers.get("content-type", "")
                if "pdf" in content_type or s3_url.lower().endswith(".pdf"):
                    raw_text = extract_text_from_pdf_bytes(r.content)
                elif s3_url.lower().endswith(".docx"):
                    raw_text = extract_text_from_docx_bytes(r.content)
                else:
                    raw_text = r.text
        except Exception as e:
            logger.exception(f"Failed to fetch from S3: {e}")
            raw_text = ""
    
    raw_text = (raw_text or "").strip()
    
    if not raw_text:
        return {
            "parsed": {
                "error": "no_text_extracted",
                "name": None,
                "skills": [],
                "summary": ""
            }
        }
    
    try:
        parsed = groq_parse_resume(raw_text)
    except Exception as e:
        logger.exception(f"Resume parsing failed: {e}")
        parsed = regex_parse_resume(raw_text)
    
    return {"parsed": parsed, "raw_text_length": len(raw_text)}

@app.get("/performance_metrics")
def get_performance_metrics(session_id: str):
    """Get current interview performance metrics (mock endpoint - would query from DB)"""
    # In production, this would query your database
    # For now, return a template
    return {
        "session_id": session_id,
        "metrics": {
            "question_count": 0,
            "average_score": 0.0,
            "trend": "unknown",
            "recommendation": "continue"
        }
    }

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Enhanced AI Interview Service",
        "model": GROQ_MODEL,
        "version": "2.0",
        "features": [
            "Deep technical question generation",
            "Multi-dimensional scoring",
            "Aggressive termination rules",
            "Bluff detection",
            "Gray-area probing",
            "Performance analytics"
        ]
    }


from scipy.spatial.distance import cosine # <--- ADD THIS IMPORT

@app.post("/verify_face")
def verify_face(req: FaceVerificationRequest):
    """
    Manual Verification: 
    1. Generates embedding for current frame.
    2. Compares it mathematically to the stored reference.
    3. Handles "No Face" gracefully without 500 Errors.
    """
    
    # 1. Decode Images
    img1 = decode_base64_image(req.reference_image) # Registration Image
    img2 = decode_base64_image(req.current_image)   # Webcam Frame
    
    if img1 is None or img2 is None:
        return JSONResponse(status_code=400, content={"verified": False, "error": "Image decode failed"})

    try:
        # 2. Get Embedding for Reference (Should be cached in reality, but calculating here for safety)
        # Note: In a real app, you'd fetch 'rep1' from your database (FACE_DB) instead of calculating it every time.
        objs1 = DeepFace.represent(
            img_path=img1,
            model_name="VGG-Face",
            detector_backend="mtcnn",
            enforce_detection=True
        )
        embedding1 = objs1[0]["embedding"]

        # 3. Get Embedding for Webcam (The critical part)
        try:
            objs2 = DeepFace.represent(
                img_path=img2,
                model_name="VGG-Face",
                detector_backend="mtcnn",
                enforce_detection=True 
            )
            embedding2 = objs2[0]["embedding"]
            
        except ValueError as e:
            # THIS catches "No Face Detected" cleanly
            logger.warning(f"No face found in webcam frame: {str(e)}")
            return JSONResponse(
                status_code=400, 
                content={
                    "verified": False, 
                    "error": "No face detected in frame. Adjust lighting.",
                    "violation_type": "no_face_detected"
                }
            )

        # 4. Calculate Distance Manually
        # Cosine Distance = 1 - Cosine Similarity
        distance = cosine(embedding1, embedding2)
        
        logger.info(f" Calculated Distance: {distance:.4f} (Threshold: {STRICT_DISTANCE_THRESHOLD})")

        # 5. Compare against Threshold
        if distance <= STRICT_DISTANCE_THRESHOLD:
            return {
                "verified": True,
                "distance": distance,
                "threshold": STRICT_DISTANCE_THRESHOLD,
                "model": "VGG-Face"
            }
        else:
            return JSONResponse(
                status_code=400, 
                content={
                    "verified": False,
                    "distance": distance,
                    "threshold": STRICT_DISTANCE_THRESHOLD,
                    "error": "Face mismatch (Unauthorized User)",
                    "violation_type": "face_mismatch"
                }
            )

    except Exception as e:
        logger.error(f"System Error in verification: {e}")
        return JSONResponse(status_code=500, content={"verified": False, "error": str(e)})
@app.get("/health")
def health():
    return {"status": "healthy", "model": GROQ_MODEL}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)