# ai-interview-ai-service/main.py
# Enhanced AI Interview Service with Strict Technical Assessment
# Features:
#  - Deep technical question generation from resume projects
#  - Advanced bluff detection and gray-area scoring
#  - Aggressive early termination for poor candidates
#  - Multi-dimensional scoring with confidence tracking
import os
import sys

# --- 1. CRITICAL: OMP FIX MUST BE FIRST ---
# This prevents the "OMP: Error #15" crash when using DeepFace + YOLO
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from groq import Groq
from typing import Tuple
import requests
import math
from dotenv import load_dotenv
from ultralytics import YOLO
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
PISTON_API_URL = "https://emkc.org/api/v2/piston/execute"
groq_client = Groq(api_key=GROQ_API_KEY)

app = FastAPI(title="Enhanced AI Interview Service")
STRICT_DISTANCE_THRESHOLD = 0.65
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-service")
print("⏳ Loading VGG-Face model... please wait...")
DeepFace.build_model("VGG-Face")
print("⏳ Loading YOLOv8 model (for phone detection)...")
object_model = YOLO("yolov8s.pt")
print("✅ All Models loaded!")
# ==========================================
# CORE CONFIGURATION
# ==========================================

TERMINATION_RULES = {
    "instant_fail_threshold": 0.20,
    "consecutive_fail_count": 2,
    "consecutive_fail_threshold": 0.45,
    "excellence_threshold": 0.85,
    "excellence_count": 3,
    
    # 👇 ADD THIS LINE (Fixes the 500 Crash) 👇
    "max_questions": 15,
    
    "min_confidence_to_end": 0.85,
    "max_questions_soft_limit": 12,
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
def scan_frame_for_violations(img_array: np.ndarray) -> Dict[str, Any]:
    """
    Scans for BOTH prohibited items and multiple people using YOLOv8.
    """
    # Run inference
    results = object_model(img_array, verbose=False, conf=0.3)
    
    detected_items = []
    person_count = 0
    
    # YOLO COCO Class IDs: 0 = Person, 67 = Cell Phone
    PROHIBITED_CLASSES = {67: "cell phone"} 
    
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            
            # Check for People
            if cls_id == 0:
                person_count += 1
            
            # Check for Objects
            if cls_id in PROHIBITED_CLASSES:
                item_name = PROHIBITED_CLASSES[cls_id]
                detected_items.append(item_name)
                
    return {
        "person_count": person_count,
        "prohibited_items": list(set(detected_items))
    }         
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
def enforce_test_cases_for_challenge(parsed: dict, resp_raw: str, original_prompt: str, max_attempts: int = 3):
    """
    Ensure parsed (which is a dict) that contains a coding_challenge ends up with:
      coding_challenge.test_cases -> list of >=2 {"input": "<json-string>", "expected": "<string>"}
    If repair succeeds, returns the updated 'challenge' dict.
    If fails after attempts, raises HTTPException(status_code=422, detail={...})
    """
    # Defensive checks
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=500, detail="internal: parsed must be dict for repair")

    challenge = parsed.get("coding_challenge") or {}
    # quick normalization: ensure legacy fields are strings if present
    if "test_case_input" in challenge and not isinstance(challenge["test_case_input"], str):
        try:
            challenge["test_case_input"] = json.dumps(challenge["test_case_input"])
        except:
            challenge["test_case_input"] = str(challenge["test_case_input"])
    if "expected_output" in challenge and not isinstance(challenge["expected_output"], str):
        try:
            challenge["expected_output"] = json.dumps(challenge["expected_output"])
        except:
            challenge["expected_output"] = str(challenge["expected_output"])

    # helper to validate normalized list
    def is_valid_tc_list(tc_list):
        if not isinstance(tc_list, list) or len(tc_list) < 2:
            return False
        for tc in tc_list:
            if not isinstance(tc, dict) or "input" not in tc or "expected" not in tc:
                return False
            if not isinstance(tc["input"], str) or not isinstance(tc["expected"], str):
                return False
        return True

    # If already good, return
    if is_valid_tc_list(challenge.get("test_cases", [])):
        # ensure legacy fields reflect first case
        first = challenge["test_cases"][0]
        challenge["test_case_input"] = first["input"]
        challenge["expected_output"] = first["expected"]
        parsed["coding_challenge"] = challenge
        return challenge

    # Build a *strict* repair prompt with original prompt + raw model output + parsed
    repair_prompt_template = """
ERROR: A coding question was generated but the required 'test_cases' array (>=2 entries) is missing or malformed.

You will be given:
1) The original human-readable generation PROMPT (below).
2) The original LLM RAW output (below).
3) The original parsed JSON object (below).

Return ONLY a single JSON object with one key "test_cases". The value MUST be an array of at least 2 test case objects. Each test case object must be exactly:
{ "input": "<valid JSON string>", "expected": "<exact expected output as string>" }

Rules (must follow exactly):
- Do not return any keys other than "test_cases".
- Do not add commentary or explanation.
- Inputs must be valid JSON strings (e.g., arrays like "[1,2,3]", strings like "\"abc\"", numbers like "5").
- If expected result is None, use "null". If boolean, use "true"/"false".
- Make tests relevant and consistent with the problem statement; include 2 different cases (e.g., typical and edge).

--- ORIGINAL PROMPT:
{orig_prompt}

--- ORIGINAL LLM RAW:
{llm_raw}

--- ORIGINAL PARSED (truncated):
{parsed_json}
""".strip()

    repair_raws = []
    for attempt in range(1, max_attempts + 1):
        repair_prompt = repair_prompt_template.format(
            orig_prompt=safe_truncate(original_prompt or "", 6000),
            llm_raw=safe_truncate(resp_raw or "", 4000),
            parsed_json=safe_truncate(json.dumps(parsed, default=str), 2000)
        )
        try:
            repair_resp = groq_call(repair_prompt, temperature=0.0, max_tokens=500)
        except Exception as e:
            repair_raws.append(f"exception:{e}")
            logger.warning("Repair groq_call exception attempt %d: %s", attempt, e)
            continue

        if not repair_resp.get("ok"):
            repair_raws.append(repair_resp.get("raw") or repair_resp.get("error") or "no_raw")
            continue

        repair_raw = repair_resp.get("raw", "")
        repair_raws.append(repair_raw[:4000])

        candidate = extract_json_from_text(repair_raw)
        if not candidate or not isinstance(candidate, dict):
            logger.info("Repair attempt %d returned non-JSON or unparsable text.", attempt)
            continue

        tc_list = candidate.get("test_cases")
        if not tc_list or not isinstance(tc_list, list) or len(tc_list) < 2:
            logger.info("Repair attempt %d returned test_cases but invalid length/format.", attempt)
            continue

        # Normalize each entry -> strings
        normalized = []
        malformed = False
        for tc in tc_list:
            if not isinstance(tc, dict) or "input" not in tc or "expected" not in tc:
                malformed = True
                break
            inp = tc["input"]
            exp = tc["expected"]
            if not isinstance(inp, str):
                try:
                    inp = json.dumps(inp)
                except:
                    inp = str(inp)
            if not isinstance(exp, str):
                try:
                    exp = json.dumps(exp)
                except:
                    exp = str(exp)
            normalized.append({"input": inp, "expected": exp})

        if malformed or len(normalized) < 2:
            logger.info("Repair attempt %d returned malformed test_cases.", attempt)
            continue

        # success
        challenge["test_cases"] = normalized
        first = normalized[0]
        challenge["test_case_input"] = first["input"]
        challenge["expected_output"] = first["expected"]
        parsed["coding_challenge"] = challenge
        parsed["_auto_repaired_test_cases"] = True
        parsed["confidence"] = min(parsed.get("confidence", 0.6), 0.6)
        logger.info("Repair succeeded on attempt %d (added %d tests).", attempt, len(normalized))
        return challenge

    # If we reach here, all attempts failed -> return structured 422 (so frontend can prompt regen)
    detail = {
        "error": "model_failed_to_provide_test_cases",
        "message": "LLM failed to return required test_cases after repair attempts.",
        "repair_raws": repair_raws[:3],
        "original_llm_raw": (resp_raw or "")[:3000]
    }
    logger.error("Test-case repair failed. Samples: %s", repair_raws[:2])
    raise HTTPException(status_code=422, detail=detail)

def build_generate_question_prompt(context: dict, mode: str = "first") -> str:
    """
    Generate a strict prompt that asks the model to produce a single JSON object
    containing a code question and exactly 3 test cases.

    IMPORTANT:
    - The JSON schema below contains an EMPTY 'test_cases' array. The model MUST
      generate 3 test cases and fill that array. Do NOT copy any example values
      that appear in the prompt.
    - The model must output EXACTLY one JSON object and NOTHING ELSE (no markdown,
      no comments, no explanatory text).
    """
    resume = context.get("resume", "")
    chunks = context.get("chunks", []) or []
    history = context.get("history", []) or []

    chunks_text = "\n".join([f"- {c.get('snippet','')[:400]}" for c in chunks[:3]])
    last_q_context = ""
    if history:
        last = history[-1]
        last_q_context = f"PREVIOUS Q: {last.get('question', '')[:150]} (Score: {last.get('score', 0)})"

    # JSON skeleton that must be filled by the model (test_cases is empty here)
    schema = '''
{
  "question": "string (a short programming prompt derived from the resume/context)",
  "type": "code",
  "expected_answer_type": "code",
  "target_project": "string (which project from resume this maps to, or 'general')",
  "difficulty": "easy|medium|hard",
  "coding_challenge": {
      "language": "python",
      "function_signature": "string (e.g. def solve(data): )",
      "starter_code": "string (short starter code; may be empty)",
      "test_cases": []
  },
  "ideal_answer_outline": "string (brief steps of solution)",
  "confidence": 0.0-1.0
}
'''.strip()

    # Helpful examples for style — THE MODEL MUST NOT COPY THESE VALUES.
    # They are shown only to demonstrate format and should be treated as examples.
    examples = '''
EXAMPLES (DO NOT COPY — for format only):
[
  {"input": "[1, 2, 3]", "expected": "6"},
  {"input": "[]", "expected": "0"},
  {"input": "[5, -1, 2]", "expected": "6"}
]
'''.strip()

    prompt = f"""
SYSTEM: You are a deterministic Question Generation Engine. You MUST output EXACTLY one valid JSON object and NOTHING ELSE (no commentary, no markdown). Follow instructions precisely.

CRITICAL INSTRUCTIONS (READ CAREFULLY):
1) Output MUST be valid JSON parsable by json.loads().
2) Output MUST follow the exact top-level schema shown below and MUST include "type": "code".
3) Under "coding_challenge", the "test_cases" array MUST contain exactly 3 test-case objects.
   Each test-case object MUST be of the form:
     {{ "input": "<valid JSON string>", "expected": "<expected output as string>" }}
   - Examples of valid JSON string inputs: "[1,2,3]", "\"abc\"", "5"
   - If expected output is null, use the string "null". Booleans must be "true"/"false".
4) DO NOT copy or echo any concrete values shown in this prompt template or the examples section.
   You MUST GENERATE NEW question text and NEW test cases based on the resume/context.
5) Inputs and expected values MUST be strings (i.e., quoted). Use json.dumps-style strings for objects/arrays.
6) Do NOT include any extra keys beyond the schema. Do NOT include comments or explanatory text.
7) Prefer problems that are solvable in a short Python function, and keep difficulty consistent with the resume's seniority (use 'easy' for junior, 'medium' for mid-level, 'hard' for senior).
8) If the resume/context indicates a specific domain (e.g., "web scraper", "trading bot"), generate a relevant problem tied to that domain.

INPUT CONTEXT:
Resume Summary:
{resume[:1200]}

Technical Docs (RAG):
{chunks_text}

Interview History:
{last_q_context}

REQUIRED JSON SCHEMA (you MUST fill this object and ONLY this object):
{schema}

FORMAT EXAMPLES FOR TEST CASE SHAPE (EXAMPLES ONLY — DO NOT COPY):
{examples}

END.
"""
    return prompt.strip()

# ==========================================
# SCORING SYSTEM
# ==========================================

def build_score_prompt(question_text: str, ideal_outline: str, candidate_answer: str, context: dict = None) -> str:
    """
    STRICT MULTI-DIMENSIONAL SCORING with advanced bluff detection.
    Now supports DUAL MODES: 
    1. Text Analysis (Concept depth, bluff detection)
    2. Code Analysis (Execution success, syntax, efficiency)
    """
    context = context or {}
    resume = context.get("resume", "")
    chunks = context.get("chunks", [])
    
    # NEW: Extract question type and execution results
    question_type = context.get("question_type", "text") 
    exec_result = context.get("code_execution_result", {})
    
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
    
    # =========================================================
    # LOGIC BRANCH: CODE vs TEXT
    # =========================================================
    
    if question_type == "code":
        # --- CODE SCORING MODE ---
        passed = exec_result.get("passed", False)
        output_log = exec_result.get("output", "No output captured")
        error_type = exec_result.get("error", "None")

        system = f"""You are a Senior Code Reviewer.
        
        CONTEXT: The candidate wrote code to solve a specific problem.
        
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        ⚠️ EXECUTION STATUS: {'✅ PASSED' if passed else '❌ FAILED'}
        ⚠️ ERROR TYPE: {error_type}
        
        EXECUTION LOG (Stdout/Stderr):
        ```
        {output_log[:800]}
        ```
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        SCORING RULES FOR CODE:
        1. **IF FAILED (passed=False)**: 
           - **MAX SCORE: 0.5**. Do not score higher even if the logic looks "okay".
           - If it's a minor syntax error (missing colon), score ~0.4.
           - If logic is completely wrong or runtime error, score < 0.3.
           
        2. **IF PASSED (passed=True)**:
           - **BASE SCORE: 0.7**.
           - **Bonus (+0.1 - 0.3)**: Efficient algorithms (Big O), clean variable naming, handling edge cases, pythonic style.
           - **Penalty (-0.1 - 0.2)**: "Spaghetti code", hardcoded values, bad variable names (e.g., 'x', 'y'), redundant logic.

        3. **BLUFF DETECTION IN CODE**:
           - Did they just `print("expected output")` to cheat the test? (Automatic 0.0)
           - Is the code obviously copied (comments don't match logic)?
           
        SCORING DIMENSIONS:
        {dimensions_text}
        """

    else:
        # --- TEXT SCORING MODE (Your Original Logic) ---
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

    # Shared Schema
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
  "rationale": "string (specific evidence for score, citing execution status if code)",
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
1. Evaluate based on the specific rules (Text vs Code).
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
def run_code_in_sandbox(language: str, code: str, stdin: str = "") -> Dict[str, Any]:
    """
    Executes code safely using the Piston Public API (or compatible runner).
    Returns a dict with detailed fields for debugging:
      {
        "success": bool,
        "output": "stdout (trimmed) or combined stdout/stderr",
        "error_type": "API Error|Compilation Error|Runtime Error|Network Error|Config Error",
        "status_code": int|None,
        "raw": raw_response_object_or_text,
        "run_stage": run_stage_dict_or_none,
        "compile_stage": compile_stage_dict_or_none
      }
    """
    # Map simple names to Piston's specific language IDs or aliases
    lang_map = {
        "python": {"language": "python", "version": "*"},
        "javascript": {"language": "javascript", "version": "*"},
        "node": {"language": "javascript", "version": "*"},
        "java": {"language": "java", "version": "*"},
        "cpp": {"language": "c++", "version": "*"},
        "c": {"language": "c", "version": "*"},
        "go": {"language": "go", "version": "*"},
    }

    config = lang_map.get(language.lower())
    if not config:
        return {
            "success": False,
            "output": f"Language '{language}' not supported. Try: python, javascript, java, cpp.",
            "error_type": "Config Error",
            "status_code": None,
            "raw": None,
            "run_stage": None,
            "compile_stage": None
        }

    payload = {
        "language": config["language"],
        "version": config["version"],
        "files": [{"content": code}],
        "stdin": stdin or "",
        # Piston fields - keep some safety margins
        "run_timeout": 10000,     # ms (10s) - adjust if you need faster/longer
        "compile_timeout": 20000  # ms (20s)
    }

    max_attempts = 3
    delay = 0.5
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.post(PISTON_API_URL, json=payload, timeout=30)
        except Exception as e:
            logger.warning("Piston request attempt %d failed: %s", attempt, e)
            last_exc = e
            time.sleep(delay)
            delay *= 2
            continue

        status = getattr(resp, "status_code", None)
        raw_text = None
        try:
            raw = resp.json()
            raw_text = raw
        except Exception:
            # not JSON
            raw_text = resp.text if hasattr(resp, "text") else str(resp)

        # Non-200 -> bubble with debug
        if status != 200:
            logger.error("Piston API returned status %s: %s", status, raw_text)
            return {
                "success": False,
                "output": "Sandbox API Unavailable",
                "error_type": "API Error",
                "status_code": status,
                "raw": raw_text,
                "run_stage": None,
                "compile_stage": None
            }

        # At this point we have a 200 and raw (maybe dict)
        data = raw if isinstance(raw_text, dict) else None

        # Defensive extraction of run/compile stages across different Piston-like shapes
        run_stage = None
        compile_stage = None
        # Typical format: {"run": {...}, "compile": {...}}
        if isinstance(data, dict):
            run_stage = data.get("run") or data.get("execution") or data.get("result") or {}
            compile_stage = data.get("compile") or data.get("compile_result") or {}
        else:
            # If data isn't a dict (rare), return raw text
            return {
                "success": False,
                "output": raw_text if isinstance(raw_text, str) else str(raw_text),
                "error_type": "API Error",
                "status_code": status,
                "raw": raw_text,
                "run_stage": None,
                "compile_stage": None
            }

        # Normalize run_stage/compile_stage to dicts
        run_stage = run_stage or {}
        compile_stage = compile_stage or {}

        # If compile stage indicates compilation error
        comp_code = compile_stage.get("code") if isinstance(compile_stage, dict) else None
        if comp_code is not None and comp_code != 0:
            out = (compile_stage.get("stderr") or compile_stage.get("stdout") or "").strip()
            return {
                "success": False,
                "output": out,
                "error_type": "Compilation Error",
                "status_code": status,
                "raw": data,
                "run_stage": run_stage,
                "compile_stage": compile_stage
            }

        # If run stage indicates runtime error (non-zero exit code)
        run_code = run_stage.get("code") if isinstance(run_stage, dict) else None
        stdout = run_stage.get("stdout") if isinstance(run_stage, dict) else None
        stderr = run_stage.get("stderr") if isinstance(run_stage, dict) else None

        # prefer combined stderr if non-empty and run failed
        if run_code is not None and run_code != 0:
            combined = ""
            if stderr:
                combined = stderr.strip()
            elif stdout:
                combined = stdout.strip()
            else:
                combined = f"Non-zero exit code: {run_code}"
            return {
                "success": False,
                "output": combined,
                "error_type": "Runtime Error",
                "status_code": status,
                "raw": data,
                "run_stage": run_stage,
                "compile_stage": compile_stage
            }

        # Success path: prefer stdout then fallback to run_stage 'output' or 'message'
        out_text = ""
        if stdout is not None and str(stdout).strip() != "":
            out_text = str(stdout).strip()
        else:
            # some runners use other keys
            possible = []
            if isinstance(run_stage, dict):
                for k in ("output", "message", "result", "stdout"):
                    v = run_stage.get(k)
                    if v:
                        possible.append(str(v).strip())
            out_text = possible[0] if possible else ""

        return {
            "success": True,
            "output": out_text,
            "error_type": None,
            "status_code": status,
            "raw": data,
            "run_stage": run_stage,
            "compile_stage": compile_stage
        }

    # If we exhausted attempts
    logger.error("Piston execution failed after %d attempts: %s", max_attempts, last_exc)
    return {
        "success": False,
        "output": str(last_exc) if last_exc else "Unknown network error",
        "error_type": "Network Error",
        "status_code": None,
        "raw": None,
        "run_stage": None,
        "compile_stage": None
    }

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
    # 👇 CHANGE: str -> Any (Fixes 422 error if conversation has numbers/nulls)
    conversation: Optional[List[Dict[str,Any]]] = [] 
    question_history: Optional[List[Dict[str,Any]]] = []
    token_budget: Optional[int] = DEFAULT_TOKEN_BUDGET
    allow_pii: Optional[bool] = False
    options: Optional[Dict[str,Any]] = {}

class FaceVerificationRequest(BaseModel):
    session_id: str
    current_image: str

class FaceRegisterRequest(BaseModel):
    sessionId: str
    image: str    

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
    
    # Coding fields
    question_type: Optional[str] = "text" 
    code_execution_result: Optional[Dict[str, Any]] = None 

class CodeSubmissionRequest(BaseModel):
    language: str           
    code: str               
    stdin: Optional[str] = "" 
    expected_output: Optional[str] = None 

class ProbeRequest(BaseModel):
    request_id: str
    session_id: str
    user_id: str
    weakness_topic: str
    prev_question: str
    prev_answer: str
    resume_summary: Optional[str] = ""
    retrieved_chunks: Optional[List[Dict[str,Any]]] = []
    # 👇 CHANGE: str -> Any
    conversation: Optional[List[Dict[str,Any]]] = [] 
    token_budget: Optional[int] = DEFAULT_TOKEN_BUDGET
    allow_pii: Optional[bool] = False
    options: Optional[Dict[str,Any]] = {}

class DecisionRequest(BaseModel):
    request_id: str
    session_id: str
    user_id: str
    resume_summary: Optional[str] = ""
    # 👇 CHANGE: str -> Any
    conversation: Optional[List[Dict[str,Any]]] = [] 
    question_history: List[Dict[str,Any]]
    retrieved_chunks: Optional[List[Dict[str,Any]]] = []
    token_budget: Optional[int] = DEFAULT_TOKEN_BUDGET
    allow_pii: Optional[bool] = False
    accept_model_final: Optional[bool] = True
# ==========================================
# API ENDPOINTS
# ==========================================

# --- ADD THIS HELPER FUNCTION ABOVE generate_question ---
def generate_missing_test_cases(question_text: str, starter_code: str) -> List[Dict[str, str]]:
    """
    Dedicated repair function. If the main prompt fails to generate tests,
    this focused prompt forces the AI to create them based on the question.
    """
    prompt = f"""
SYSTEM: You are a QA Engineer.
TASK: Generate 3 strict JSON test cases for this coding problem.

PROBLEM:
{question_text}

CODE STUB:
{starter_code}

OUTPUT FORMAT:
Return ONLY a raw JSON list of objects. No markdown.
[
  {{"input": "valid_input_string", "expected": "valid_output_string"}},
  {{"input": "edge_case_input", "expected": "edge_case_output"}}
]
"""
    try:
        # Fast, low-temp call
        resp = groq_call(prompt, temperature=0.0, max_tokens=300)
        cases = extract_json_from_text(resp.get("raw", ""))
        
        valid = []
        if isinstance(cases, list):
            for c in cases:
                if isinstance(c, dict) and "input" in c and "expected" in c:
                    valid.append({
                        "input": str(c["input"]), 
                        "expected": str(c["expected"])
                    })
        return valid
    except:
        return []

# --- REPLACE THE MAIN ENDPOINT ---
@app.post("/generate_question")
def generate_question(req: GenerateQuestionRequest):
    payload = req.dict()
    enforced = enforce_budget(payload)
    enforced["history"] = payload.get("question_history", [])

    # Prepare prompt once (deterministic)
    prompt = build_generate_question_prompt(enforced, mode=payload.get("mode", "first"))

    # Build ordered list of models to try
    tried_models = []
    default_model = os.getenv("GROQ_MODEL", GROQ_MODEL)
    tried_models.append(default_model)
    alt_models_env = os.getenv("ALT_GROQ_MODELS", "")
    if alt_models_env:
        for m in [m.strip() for m in alt_models_env.split(",") if m.strip()]:
            if m not in tried_models:
                tried_models.append(m)

    generation_attempts = []
    parsed = None
    chosen_raw = None
    chosen_model = None

    try:
        for model_name in tried_models:
            logger.info("generate_question: trying model '%s' for request_id=%s", model_name, payload.get("request_id"))
            try:
                resp = groq_call(prompt, model=model_name, temperature=0.0, max_tokens=1200)
            except Exception as e:
                logger.exception("groq_call failed for model %s: %s", model_name, e)
                generation_attempts.append({"model": model_name, "raw": None, "error": str(e)})
                continue

            raw = resp.get("raw", "") if isinstance(resp, dict) else str(resp)
            generation_attempts.append({"model": model_name, "raw": raw})

            candidate = extract_json_from_text(raw)
            if not candidate or not isinstance(candidate, dict):
                logger.info("Model '%s' returned unparsable or non-dict JSON; continuing.", model_name)
                continue

            # Validate it's a code question with >=3 properly shaped test cases
            if candidate.get("type") != "code":
                logger.info("Model '%s' returned non-code type; continuing.", model_name)
                continue

            cc = candidate.get("coding_challenge") or {}
            tcs = cc.get("test_cases")
            def valid_tc_list(tc_list):
                if not isinstance(tc_list, list) or len(tc_list) < 3:
                    return False
                for tc in tc_list:
                    if not isinstance(tc, dict):
                        return False
                    if "input" not in tc or "expected" not in tc:
                        return False
                    if not isinstance(tc["input"], str) or not isinstance(tc["expected"], str):
                        return False
                return True

            if valid_tc_list(tcs):
                # success
                parsed = candidate
                chosen_raw = raw
                chosen_model = model_name
                logger.info("Model '%s' produced valid test_cases for request_id=%s", model_name, payload.get("request_id"))
                break
            else:
                logger.info("Model '%s' produced invalid or insufficient test_cases; continuing.", model_name)

        # If no model produced a valid parsed result, fail loudly with debugging info
        if parsed is None:
            logger.error("All attempted models failed to produce valid test_cases for request_id=%s", payload.get("request_id"))
            raise HTTPException(status_code=502, detail={
                "error": "model_failed_to_provide_test_cases",
                "message": "None of the attempted models returned a coding challenge with >=3 valid test_cases.",
                "attempts": generation_attempts
            })

    except HTTPException:
        # Re-raise HTTPException so FastAPI handles properly
        raise
    except Exception as e:
        logger.exception("Unexpected generation error: %s", e)
        raise HTTPException(status_code=500, detail="AI generation failed")

    # At this point 'parsed' is valid and contains coding_challenge.test_cases >= 3
    # Optionally, enforce legacy fields for frontend compatibility
    try:
        challenge = parsed.get("coding_challenge") or {}
        tcs = challenge.get("test_cases", [])
        if isinstance(tcs, list) and len(tcs) >= 1:
            challenge["test_case_input"] = tcs[0]["input"]
            challenge["expected_output"] = tcs[0]["expected"]
        if "language" not in challenge:
            challenge["language"] = "python"
        if "starter_code" not in challenge:
            challenge["starter_code"] = "def solve(x):\n    pass"
        parsed["coding_challenge"] = challenge
    except Exception:
        logger.warning("Failed to normalize legacy coding_challenge fields; continuing with parsed result.")

    return {
        "request_id": payload["request_id"],
        "llm_raw": chosen_raw,
        "parsed": parsed,
        "redaction_log": []
    }

@app.post("/run_code")
def run_code(req: CodeSubmissionRequest):
    final_code = req.code

    # ---------------------------------------------------------
    # 1. ROBUST DRIVER (Reads from Sys.Stdin)
    # ---------------------------------------------------------
    # Inject driver if Python and file looks like a library (has a def
    # but no "__main__" execution block). Avoid relying on "print(" check.
    if req.language.lower() == "python" and "def " in final_code and "if __name__" not in final_code:

        # Regex to find the function name
        target_func = "solve"
        match = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", final_code)
        if match:
            target_func = match.group(1)

        # Universal driver: safe parsing + fallback call strategies
        driver = f'''
import sys, json, traceback

def _parse_input(raw):
    raw = raw.strip()
    if raw == "":
        return None
    # Try JSON first
    try:
        return json.loads(raw)
    except:
        pass
    # If looks like CSV without brackets: "1, 2, 3"
    if ',' in raw and '[' not in raw and raw.count('"') % 2 == 0:
        parts = [p.strip() for p in raw.split(',')]
        # Try convert to ints/floats, otherwise keep strings
        out = []
        for p in parts:
            if p == "":
                continue
            try:
                out.append(int(p))
                continue
            except:
                pass
            try:
                out.append(float(p))
                continue
            except:
                pass
            # strip quotes if present
            if len(p) >= 2 and ((p[0] == p[-1] == '"') or (p[0] == p[-1] == "'")):
                out.append(p[1:-1])
            else:
                out.append(p)
        return out
    # Fallback: raw string (strip wrapping quotes if present)
    if len(raw) >= 2 and ((raw[0] == raw[-1] == '"') or (raw[0] == raw[-1] == "'")):
        return raw[1:-1]
    return raw

if __name__ == "__main__":
    try:
        raw_input = sys.stdin.read()
        input_data = _parse_input(raw_input)

        # Try calling with one argument, fallback to zero-arg if TypeError
        result = None
        try:
            result = {target_func}(input_data)
        except TypeError:
            try:
                result = {target_func}()
            except Exception as e:
                # re-raise to be caught below
                raise

        # Print result in JSON-friendly form
        if result is None:
            print("null")
        else:
            try:
                print(json.dumps(result))
            except TypeError:
                # Objects not JSON serializable -> fallback to str()
                print(json.dumps(str(result)))
    except Exception as e:
        # Provide structured driver error for debugging
        tb = traceback.format_exc()
        print("DRIVER_ERROR")
        print(json.dumps({{"error": str(e), "traceback": tb}}))
'''
        final_code = final_code + "\n\n" + driver

    # ---------------------------------------------------------
    # 2. EXECUTION (Pass stdin normally)
    # ---------------------------------------------------------
    run_result = run_code_in_sandbox(req.language, final_code, req.stdin)

    # Build base response (include lots of debug info)
    response = {
        "success": run_result.get("success", False),
        "output": str(run_result.get("output", "") or "").strip(),
        "error": run_result.get("error_type"),
        "passed": False,
        "stdin_received": req.stdin or "",
        "raw_run": run_result,   # full raw sandbox response for troubleshooting
        "debug": None
    }

    # ---------------------------------------------------------
    # 3. ROBUST GRADING
    # ---------------------------------------------------------
    if req.expected_output and run_result.get("success"):
        actual = response["output"]
        expected = str(req.expected_output).strip()

        # Exact match
        if actual == expected:
            response["passed"] = True
        else:
            # Try JSON decode both sides (handles spacing, ordering for arrays/objects)
            try:
                actual_json = json.loads(actual) if actual not in ("None", "") else None
            except:
                actual_json = None
            try:
                expected_json = json.loads(expected) if expected not in ("None", "") else None
            except:
                expected_json = None

            if actual_json is not None and expected_json is not None:
                if actual_json == expected_json:
                    response["passed"] = True

        if not response["passed"]:
            response["debug"] = f"Expected: {expected} | Got: {actual}"
            logger.info(f"❌ TEST FAILED: {response['debug']}")

    # If sandbox failed, include its message in debug for easier triage
    if not run_result.get("success"):
        response["debug"] = response.get("debug") or run_result.get("output") or run_result.get("error_type")

    return response

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
        "chunks": enforced.get("chunks", []),
        "question_type": payload.get("question_type", "text"),          # <--- ADDED
        "code_execution_result": payload.get("code_execution_result")   # <--- ADDED
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
    MTCNN Verification:
    - Uses 'mtcnn' backend: A 3-stage deep learning detector.
    - Stage 3 explicitly verifies facial landmarks, so notebooks/books are rejected.
    - More reliable than OpenCV, faster than RetinaFace.
    """
    # 1. Validate Session
    if req.session_id not in FACE_DB:
        return JSONResponse(status_code=400, content={"verified": False, "error": "Session not found."})

    reference_embedding = FACE_DB[req.session_id]["embedding"]

    # 2. Decode image
    img2 = decode_base64_image(req.current_image)
    if img2 is None:
        return JSONResponse(status_code=400, content={"verified": False, "error": "Image decode failed"})

    # =========================================================================
    # CHECK 1: COUNT FACES with MTCNN
    # =========================================================================
    # MTCNN is very strict. It requires eyes/nose/mouth to confirm a face.
    try:
        face_objs = DeepFace.extract_faces(
            img_path=img2,
            detector_backend="mtcnn",   # <--- SWAPPED TO MTCNN
            enforce_detection=False,    # Don't crash if 0 faces
            align=True
        )
        
        # MTCNN confidence scores are usually very reliable.
        # We filter out anything below 0.8 to be safe against shadows.
        valid_faces = [f for f in face_objs if f.get('confidence', 0) > 0.80]
        face_count = len(valid_faces)
        
    except Exception as e:
        logger.error(f"MTCNN error: {e}")
        # If MTCNN fails (e.g. image too small), we default to 0 to be safe
        face_count = 0

    # =========================================================================
    # CHECK 2: PROHIBITED OBJECTS (YOLO Only)
    # =========================================================================
    detected_items = []
    try:
        # We ONLY use YOLO for phones, not for counting people/faces
        results = object_model(img2, verbose=False, conf=0.40)
        PROHIBITED_CLASSES = {67: "cell phone"} 
        
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id in PROHIBITED_CLASSES:
                    detected_items.append(PROHIBITED_CLASSES[cls_id])
                    
    except Exception as e:
        logger.warning(f"Object detection skipped: {e}")

    # =========================================================================
    # DECISION LOGIC
    # =========================================================================
    
    # 1. Check Multiple People
    if face_count > 1:
        return JSONResponse(
            status_code=400,
            content={
                "verified": False,
                "violation_type": "multiple_people",
                "error": "Multiple people detected",
                "person_count": face_count,
                "details": f"MTCNN detected {face_count} distinct faces."
            }
        )
    
    # 2. Check Objects
    if detected_items:
        return JSONResponse(status_code=400, content={"verified": False, "violation_type": "prohibited_object", "objects": detected_items, "error": "Prohibited object detected"})

    # 3. Check No Face
    if face_count == 0:
         return JSONResponse(status_code=400, content={"verified": False, "error": "No face detected", "violation_type": "no_face_detected"})

    # =========================================================================
    # CHECK 3: IDENTITY MATCH (VGG-Face via MTCNN)
    # =========================================================================
    try:
        # We use the same backend to ensure alignment consistency
        # We assume the valid face found above is the one we want to check
        embedding_objs = DeepFace.represent(
            img_path=img2,
            model_name="VGG-Face",
            detector_backend="mtcnn",
            enforce_detection=True
        )
        
        # Take the most confident face (DeepFace usually sorts by size/confidence)
        current_embedding = embedding_objs[0]["embedding"]
        
        distance = cosine(reference_embedding, current_embedding)
        
        if distance <= STRICT_DISTANCE_THRESHOLD:
            return {"verified": True, "distance": distance}
        else:
            return JSONResponse(status_code=400, content={"verified": False, "distance": distance, "error": "Face mismatch", "violation_type": "face_mismatch"})

    except Exception as e:
        # If verify fails despite finding a face earlier, it's usually an alignment edge case
        return JSONResponse(status_code=500, content={"verified": False, "error": f"Identity check failed: {str(e)}"})
def health():
    return {"status": "healthy", "model": GROQ_MODEL}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)