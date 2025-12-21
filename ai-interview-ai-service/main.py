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
from pydantic import BaseModel, ConfigDict
from typing import Any, Dict, List, Optional

from groq import Groq
from google import genai
from google.genai import types
from openai import OpenAI

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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-service")

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY not set")
    
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    logger.warning("GROQ_API_KEY not set. Groq-based parsing will fail.")
    groq_client = None
# 2. Google GenAI (Specific for Resume Parsing - Optional but FREE)
# If you don't have this key, the resume parser will fallback to OpenRouter
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
try:
    google_client = genai.Client(api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None
except:
    google_client = None
    
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")  # Using most capable model
MAX_PROMPT_CHARS = int(os.getenv("MAX_PROMPT_CHARS", "14000"))
DEFAULT_TOKEN_BUDGET = int(os.getenv("DEFAULT_TOKEN_BUDGET", "5000"))
PISTON_API_URL = "https://emkc.org/api/v2/piston/execute"

app = FastAPI(title="Enhanced AI Interview Service")
STRICT_DISTANCE_THRESHOLD = 0.65
print("⏳ Loading VGG-Face model... please wait...")
DeepFace.build_model("VGG-Face")
print("⏳ Loading YOLOv8 model (for phone detection)...")
object_model = YOLO("yolov8s.pt")
print("✅ All Models loaded!")
RESUME_PARSER_CHAIN = [
    {"provider": "groq", "model": "llama-3.3-70b-versatile"},
    {"provider": "groq", "model": "mixtral-8x7b-32768"}
]

# ==========================================
# CORE CONFIGURATION
# ==========================================

# USE THE NEW ALIVE MODEL (1.5 is dead)
INTERVIEW_MODELS = [
    # Primary: Fast, reliable, great instruction following
    "meta-llama/llama-3.3-70b-instruct",
    
    # Specialist: Excellent at coding/logic (use for generate_question)
    "qwen/qwen-2.5-coder-32b-instruct",
    
    # Reasoner: Good for scoring/decisions (DeepSeek R1 Distill)
    "deepseek/deepseek-r1-distill-llama-70b",
]
INTERVIEW_FLOW = [
    "project_discussion",  # Q1: Warm up with their best project
    "experience",          # Q2: Verify work history (or Core Concept if fresher)
    "coding_challenge",    # Q3: DSA Check 1 (Arrays/Strings)
    "project_discussion",  # Q4: Deep dive into a DIFFERENT project
    "coding_challenge",    # Q5: DSA Check 2 (Trees/HashMaps)
    "achievement",         # Q6: Behavioral/Achievement check
    "coding_challenge",    # Q7: Final Complexity Check (Optional)
    "conceptual"           # Q8+: System Design / filler
]
# ==========================================
# 1. Update INTERVIEW_ROUNDS
# ==========================================
# ==========================================
# 1. Update INTERVIEW_ROUNDS
# ==========================================
INTERVIEW_ROUNDS = {
    "screening": {
        "name": "Screening Round",
        "focus": ["experience", "resume_basics", "fundamentals"],
        "min_questions": 1,
        "max_questions": 2,
        "pass_threshold": 0.55,
        "elimination": True
    },
    "technical": {
        "name": "Technical Round",
        "focus": ["project_discussion", "coding_challenge", "system_design"], # <--- Included here
        "min_questions": 3, 
        "max_questions": 5, # Increased max to allow for a design question
        "pass_threshold": 0.60,
        "elimination": True
    },
    "behavioral": {
        "name": "Behavioral Round",
        "focus": ["achievement", "collaboration", "ownership"],
        "min_questions": 1,
        "max_questions": 2,
        "pass_threshold": 0.5,
        "elimination": True
    }
}
# ==========================================
# 2. Update QUESTION_TYPE_TO_ROUND
# ==========================================
QUESTION_TYPE_TO_ROUND = {
    # --- Screening (Experience & Basics) ---
    "experience": "screening",
    "resume_basics": "screening",
    "fundamentals": "screening",
    "conceptual": "screening",  # Fallback

    # --- Technical (Projects & Coding) ---
    "project_discussion": "technical",  # <--- Moved to Technical
    "coding_challenge": "technical",
    "system_design": "technical",
        "debugging": "technical",


    # --- Behavioral (Achievements & HR) ---
    "achievement": "behavioral",
    "ownership": "behavioral",
    "collaboration": "behavioral",
    "reflection": "behavioral"
}

TERMINATION_RULES = {
    "instant_fail_threshold": 0.20,
    "consecutive_fail_count": 2,
    "consecutive_fail_threshold": 0.45,
    "excellence_threshold": 0.85,
    "excellence_count": 3,
    
    # 👇 ADD THIS LINE (Fixes the 500 Crash) 👇
    "max_questions": 10,

    "min_confidence_to_end": 0.85,
    "max_questions_soft_limit": 9,
    "gray_zone_min": 0.40,
    "gray_zone_max": 0.75,
}
# ==========================================
# 🛑 EMERGENCY FALLBACK QUESTIONS (PREVENTS CRASHES)
# ==========================================
FALLBACK_QUESTIONS = {
    "coding_challenge": {
        "question": "Write a Python function to check if a string is a valid palindrome, considering only alphanumeric characters and ignoring case.",
        "type": "coding_challenge",
        "coding_challenge": {
            "language": "python",
            "starter_code": "def is_palindrome(s):\n    pass",
            "test_cases": [
                {"input": "\"A man, a plan, a canal: Panama\"", "expected": "true"},
                {"input": "\"race a car\"", "expected": "false"}
            ]
        }
    },
    "project_discussion": {
        "question": "Can you walk me through the most challenging bug you encountered in your recent project? How did you debug and resolve it?",
        "type": "project_discussion"
    },
    "conceptual": {
        "question": "Explain the difference between a process and a thread in an operating system. When would you choose one over the other?",
        "type": "conceptual"
    }
}
INTERVIEW_MODE = {
    "role": "campus_recruiter",
    "target_level": "intern_or_fresher",  # intern | fresher | experienced
    "focus": [
        "resume_projects",
        "core_cs_fundamentals",
        "practical_implementation"
    ]
}
SCORING_DIMENSIONS = {
    "technical_accuracy": {
        "weight": 0.50,  # ⬆️ Increased from 0.40
        "description": "Correctness of code, logic, complexity analysis, and algorithms"
    },
    "depth_of_understanding": {
        "weight": 0.30,
        "description": "Ability to explain 'why' (trade-offs, edge cases, optimizations)"
    },
    "practical_experience": {
        "weight": 0.15,
        "description": "Evidence of real implementation, debugging, and clean code standards"
    },
    "communication_clarity": {
        "weight": 0.05,  # ⬇️ Reduced from 0.10 (Behavioral impact lowered)
        "description": "Ability to articulate ideas (less critical than code correctness)"
    }
}
# ALLOWED_GROQ_MODELS = [
#    "llama-3.3-70b-versatile"

# ]

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
# ================= SMART PROJECT INTELLIGENCE =================

def normalize_text(text: str) -> str:
    return re.sub(r'[^a-z0-9 ]', '', text.lower()).strip()

def compute_similarity(text1: str, text2: str) -> float:
    """Jaccard similarity for semantic deduplication"""
    w1 = set(normalize_text(text1).split())
    w2 = set(normalize_text(text2).split())
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)
def get_diverse_question_hint(history: List[Dict[str, Any]], required_type: str) -> str:
    """
    Generates STRICT constraints to prevent topic repetition.
    NOW: Detects probe patterns and FORCES topic switch after 1 probe.
    """
    if not history:
        return "STARTING INTERVIEW: Scan the resume. Identify the MOST COMPLEX project or skill listed and start there."
    
    # ==============================================================================
    # 1. ENHANCED STAGNATION DETECTION (CATCHES PROBE LOOPS)
    # ==============================================================================
    pivot_instruction = ""
    
    # A. Detect Probe Pattern (Last 2 questions are similar)
    if len(history) >= 2:
        last_q = history[-1].get("question", "")
        prev_q = history[-2].get("question", "")
        
        similarity = compute_similarity(last_q, prev_q)
        
        # If similarity > 0.25, we JUST probed. MANDATORY SWITCH.
        if similarity > 0.25:
            pivot_instruction += f"""
🚨 PROBE LOOP DETECTED (Similarity: {similarity:.2f})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MANDATORY ACTION: COMPLETE TOPIC SWITCH
- The last question was ALREADY a follow-up/probe
- You MUST ask about a COMPLETELY DIFFERENT topic now
- Switch category entirely (e.g., Project → DSA, DSA → Experience)
- DO NOT reference anything from the last 2 questions
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

    # B. Detect Chronic Weakness (Last answer scored low)
    last_entry = history[-1]
    last_score = last_entry.get("score")
    
    if last_score is not None and float(last_score) < 0.50:
        pivot_instruction += f"""
⚠️ LOW SCORE DETECTED ({float(last_score):.2f})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Candidate struggled with this topic
- Give them a FRESH START on a different area
- DO NOT drill deeper into this weakness
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

    # ==============================================================================
    # 2. EXISTING LOGIC: EXTRACT FORBIDDEN WORDS (Preserved)
    # ==============================================================================
    recent_words = set()
    recent_questions_text = []
    
    for h in history[-6:]:
        q = h.get("question", "").lower()
        recent_questions_text.append(q[:100])
        
        words = re.findall(r'\b[a-z]{5,}\b', q)
        stopwords = {'would', 'could', 'should', 'please', 'explain', 'describe', 
                     'implement', 'create', 'write', 'function', 'method', 'class', 
                     'what', 'how', 'when', 'why'}
        technical_words = [w for w in words if w not in stopwords]
        recent_words.update(technical_words[:15])
    
    tech_patterns = [
        r'\b(python|java|javascript|react|node|sql|aws|docker|kubernetes)\b',
        r'\b(machine learning|deep learning|neural network|nlp|cnn|lstm)\b',
        r'\b(api|rest|graphql|microservices|database|optimization)\b',
        r'\b(algorithm|data structure|sorting|searching|dynamic programming)\b'
    ]
    
    mentioned_techs = set()
    combined_history = " ".join(recent_questions_text)
    for pattern in tech_patterns:
        matches = re.findall(pattern, combined_history)
        mentioned_techs.update(matches)
    
    forbidden_list = list(recent_words)[:12] + list(mentioned_techs)[:8]
    
    hint = f"\n⛔ ABSOLUTE PROHIBITION - DO NOT ASK ABOUT:\n"
    hint += f"   Topics: {', '.join(forbidden_list)}\n"
    hint += f"   Context: These were ALREADY covered in previous questions.\n"
    
    # ==============================================================================
    # 3. MERGE: PREPEND PIVOT INSTRUCTION (Most Important)
    # ==============================================================================
    hint = pivot_instruction + hint

    # ==============================================================================
    # 4. TYPE CONSTRAINTS (Preserved)
    # ==============================================================================
    if required_type == "project_discussion":
        hint += "\n🎯 MANDATORY REQUIREMENTS:\n"
        hint += "   1. Pick a DIFFERENT project than previously discussed\n"
        hint += "   2. Focus on a SPECIFIC feature, metric, or technical decision\n"
        hint += "   3. Ask 'HOW did you implement X?' not 'What is X?'\n"
        hint += "   4. Reference exact tools/frameworks from the resume\n"
    
    elif required_type == "coding_challenge":
        hint += "\n🎯 MANDATORY REQUIREMENTS:\n"
        hint += "   1. Create a problem using a DIFFERENT algorithm/data structure\n"
        hint += "   2. Avoid problems similar to what was already asked\n"
        hint += "   3. Focus on: Arrays, Strings, HashMaps, Trees, or Graphs\n"
        hint += "   4. Make it solvable in 10-15 minutes\n"
    
    elif required_type == "conceptual":
        hint += "\n🎯 MANDATORY REQUIREMENTS:\n"
        hint += "   1. Ask about a DIFFERENT technical concept\n"
        hint += "   2. Focus on trade-offs or architecture decisions\n"
        hint += "   3. Relate to their resume but use NEW terminology\n"
    
    return hint
  
def is_repetitive_question(new_q: str, history: List[Dict[str, Any]]) -> bool:
    """🔧 IMPROVED: Better repetition detection with multiple checks"""
    if not new_q or not history:
        return False

    new_norm = normalize_text(new_q)
    new_words = set(new_norm.split())
    
    # Check last 8 questions (increased from 6)
    for h in history[-8:]:
        prev = h.get("question", "")
        prev_norm = normalize_text(prev)
        prev_words = set(prev_norm.split())

        # Exact match
        if new_norm == prev_norm:
            logger.warning(f"Exact match detected: '{new_q[:50]}...'")
            return True

        # High semantic overlap (lowered threshold from 0.70 to 0.65)
        similarity = compute_similarity(new_q, prev)
        if similarity > 0.65:
            logger.warning(f"High similarity ({similarity:.2f}): '{new_q[:50]}...'")
            return True
        
        # Check if >70% of key words are shared (new check)
        if len(new_words) > 0 and len(new_words & prev_words) / len(new_words) > 0.70:
            logger.warning(f"Word overlap detected: '{new_q[:50]}...'")
            return True

    return False

def extract_question_topics(question: str) -> set:
    """Extract key technical topics from a question for comparison"""
    q_lower = question.lower()
    
    # Technical terms to extract
    tech_terms = set()
    
    # Programming languages
    langs = ['python', 'java', 'javascript', 'typescript', 'c++', 'go', 'rust']
    for lang in langs:
        if lang in q_lower:
            tech_terms.add(lang)
    
    # Frameworks/Libraries
    frameworks = ['react', 'angular', 'vue', 'django', 'flask', 'spring', 'express', 'fastapi']
    for fw in frameworks:
        if fw in q_lower:
            tech_terms.add(fw)
    
    # Concepts
    concepts = ['api', 'database', 'optimization', 'algorithm', 'data structure', 
                'machine learning', 'neural network', 'microservices', 'docker', 'kubernetes']
    for concept in concepts:
        if concept in q_lower:
            tech_terms.add(concept.replace(' ', '_'))
    
    # Algorithm types
    algos = ['sorting', 'searching', 'tree', 'graph', 'array', 'string', 'hash', 'dynamic programming']
    for algo in algos:
        if algo in q_lower:
            tech_terms.add(algo)
    
    return tech_terms
def normalize_project_name(name: str) -> str:
    if not name:
        return ""
    name = name.lower()
    name = re.sub(r'[^a-z0-9 ]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name
    
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
def calculate_resume_coverage(projects: List[Dict], history: List[Dict]) -> Dict[str, Any]:
    coverage = {p["project_id"]: False for p in projects}

    for h in history:
        pid = h.get("target_project")

        if pid and pid in coverage:
            coverage[pid] = True

    return {
        "total": len(coverage),
        "covered": sum(coverage.values()),
        "uncovered_projects": [
            p["title"] for p in projects if not coverage[p["project_id"]]
        ]
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
def normalize_overall_score(validated: dict, history: List[Dict[str, Any]]) -> float:
    """
    Stabilizes scoring across interview.
    - Penalizes repeated weakness
    - Rewards fast improvement
    - Prevents noisy LLM jumps
    """
    score = validated.get("overall_score")
    if score is None:
        return 0.0

    score = float(score)

    # Penalize repeated weak answers
    recent = history[-3:]
    weak_count = sum(1 for h in recent if h.get("score", 1) < 0.4)
    if weak_count >= 2:
        score -= 0.10

    # Reward learning curve
    if history:
        prev = history[-1].get("score")
        if prev is not None and score > prev + 0.20:
            score += 0.05

    # Penalize low confidence mismatch
    confidence = validated.get("confidence", 0.5)
    if confidence < 0.35:
        score -= 0.05

    return max(0.0, min(1.0, score))
# Locate the existing function and REPLACE it with this robust version
def extract_whiteboard_keywords(scene_elements: List[Dict[str, Any]]) -> str:
    """
    Parses Excalidraw JSON to find what the user wrote.
    CRITICAL: Filters out deleted elements so AI doesn't see erased text.
    """
    if not scene_elements:
        return "Whiteboard is empty."

    found_text = []
    
    for el in scene_elements:
        # 1. Skip deleted elements (Excalidraw soft-deletes)
        if el.get("isDeleted", False):
            continue

        # 2. Extract text from 'text' elements
        if el.get("type") == "text" and "text" in el:
            clean_txt = el["text"].strip()
            if len(clean_txt) > 1: 
                found_text.append(clean_txt)
                
        # 3. Extract text from labeled arrows/shapes
        if "label" in el and el["label"] and "text" in el["label"]:
             clean_txt = str(el["label"]["text"]).strip()
             if len(clean_txt) > 1:
                found_text.append(clean_txt)

    if not found_text:
        return "Whiteboard contains shapes but NO text labels."

    return "Items drawn on whiteboard: " + ", ".join(found_text)
def derive_verdict_from_score(score: float) -> str:
    """
    Calibrated grading scale for technical interviews.
    Previous scale was too harsh (0.85+ for Strong).
    """
    if score < 0.35:
        return "fail"
    if score < 0.55:
        return "weak"
    if score < 0.75:
        return "acceptable" # 55% - 74% is passing for an Intern
    if score < 0.90:
        return "strong"     # 75% - 89% is Strong
    return "exceptional"    # 90%+ is Exceptional

def llm_call(prompt: str, temperature=0.3, max_tokens=1200) -> dict:
    """
    Robust LLM caller that cycles through the INTERVIEW_MODELS list.
    """
    for model in INTERVIEW_MODELS:
        try:
            resp = openrouter_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                extra_headers={
                    "HTTP-Referer": "http://localhost",
                    "X-Title": "ai-interview-assistant"
                }
            )

            raw = resp.choices[0].message.content
            if raw:
                return {
                    "ok": True,
                    "raw": raw,
                    "model_used": model
                }

        except Exception as e:
            logger.warning(f"Model {model} failed: {e}")
            # Continue to next model in list

    # If we reach here, all models failed
    return {"ok": False, "error": "all_models_failed"}


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
    BALANCED termination logic for campus / fresher interviews.
    Avoids early over-rejection while preventing endless interviews.
    """
    if not history or len(history) < 2:
        return None  # Need at least 2 questions

    metrics = calculate_performance_metrics(history)
    rules = TERMINATION_RULES

    qn = metrics["question_count"]
    avg = metrics["average_score"]
    last = metrics["last_score"]

    # --------------------------------------------------
    # RULE 1: Catastrophic Failure (Immediate Reject)
    # --------------------------------------------------
    # Candidate clearly doesn't know their own resume
    if last is not None and last < rules["instant_fail_threshold"]:
        return {
            "ended": True,
            "verdict": "reject",
            "confidence": 0.95,
            "reason": f"Catastrophic failure (score {last:.2f}). Candidate lacks basic understanding.",
            "recommended_role": None,
            "trigger": "instant_fail"
        }

    # --------------------------------------------------
    # RULE 2: Consecutive Failures (Efficiency Reject)
    # --------------------------------------------------
    if metrics["consecutive_fails"] >= rules["consecutive_fail_count"]:
        return {
            "ended": True,
            "verdict": "reject",
            "confidence": 0.90,
            "reason": f"Failed {metrics['consecutive_fails']} consecutive questions.",
            "recommended_role": None,
            "trigger": "consecutive_fails"
        }

    # --------------------------------------------------
    # RULE 3: Clear Excellence (Early Hire)
    # --------------------------------------------------
    if (
        qn >= 6 and
        avg >= rules["excellence_threshold"] and
        metrics["consecutive_wins"] >= 3
    ):
        return {
            "ended": True,
            "verdict": "hisre",
            "confidence": 0.95,
            "reason": f"Consistent excellence across {qn} questions.",
            "recommended_role": "SDE-1 / Entry-Level",
            "trigger": "proven_excellence"
        }

    # --------------------------------------------------
    # RULE 4: Chronic Weakness (Fair Reject)
    # --------------------------------------------------
    if qn >= 5:
        weak_answers = sum(1 for h in history if h.get("score", 1) < 0.45)
        weak_ratio = weak_answers / qn

        if weak_ratio >= 0.70:
            return {
                "ended": True,
                "verdict": "reject",
                "confidence": 0.88,
                "reason": f"Below minimum bar in {weak_answers}/{qn} questions.",
                "recommended_role": None,
                "trigger": "chronic_underperformance"
            }

    # --------------------------------------------------
    # RULE 5: Gray-Zone Timeout (Make a Call)
    # --------------------------------------------------
    if qn >= 7:
        gray_answers = sum(
            1 for h in history
            if rules["gray_zone_min"] <= h.get("score", 0) <= rules["gray_zone_max"]
        )

        if gray_answers >= 4:
            verdict = "hire" if avg >= 0.58 else "reject"
            return {
                "ended": True,
                "verdict": verdict,
                "confidence": 0.70,
                "reason": (
                    f"Prolonged gray-zone performance. "
                    f"Final decision based on average score {avg:.2f}."
                ),
                "recommended_role": "Intern" if verdict == "hire" else None,
                "trigger": "gray_zone_timeout"
            }

    # --------------------------------------------------
    # RULE 6: Hard Safety Limit (Never Infinite)
    # --------------------------------------------------
    if qn >= rules.get("max_questions", 12):
        verdict = "hire" if avg >= 0.60 else "reject"
        return {
            "ended": True,
            "verdict": verdict,
            "confidence": 0.80,
            "reason": "Interview reached maximum allowed questions.",
            "recommended_role": "Entry-Level" if verdict == "hire" else None,
            "trigger": "max_questions"
        }

    return None  # Continue interview

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
            title = line.strip()[:100]
            current_project = {
                "title": title,
                "project_id": normalize_project_name(title),
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
def classify_project_domain(text: str) -> str:
    DOMAIN_SIGNALS = {
        'ml_ai': ['machine learning', 'neural network', 'nlp', 'vision'],
        'web_dev': ['api', 'frontend', 'backend', 'rest', 'graphql'],
        'data_engineering': ['etl', 'pipeline', 'spark', 'airflow'],
        'devops': ['docker', 'kubernetes', 'ci/cd', 'terraform'],
        'mobile': ['android', 'ios', 'flutter'],
    }

    text = text.lower()
    scores = {k: sum(kw in text for kw in v) for k, v in DOMAIN_SIGNALS.items()}
    return max(scores, key=scores.get) if max(scores.values()) > 0 else "general"


def extract_action_keywords(text: str) -> List[str]:
    ACTIONS = [
        'implemented', 'developed', 'built', 'designed',
        'optimized', 'scaled', 'deployed', 'debugged'
    ]
    t = text.lower()
    return [a for a in ACTIONS if a in t]


def calculate_project_complexity(project: Dict) -> float:
    score = 0.0
    score += min(len(project.get("technologies", [])) / 10, 0.4)
    score += min(len(project.get("description", "").split()) / 120, 0.3)
    score += min(len(project.get("keywords", [])) / 8, 0.3)
    return round(min(score, 1.0), 2)
def validate_and_fix_test_cases(
    test_cases: List[Dict[str, str]],
    reference_code: str,
    language: str = "python"
) -> List[Dict[str, str]]:
    """
    HARD VALIDATION:
    - Execute reference code for each test case
    - Replace expected output with execution output
    - Drop invalid test cases
    """

    validated = []

    for tc in test_cases:
        raw_input = tc.get("input", "")
        if raw_input is None:
            continue

        # 🔥 Execute reference solution
        result = run_code_in_sandbox(
            language=language,
            code=reference_code,
            stdin=raw_input
        )

        # ❌ If execution failed → INVALID TEST CASE
        if not result.get("success"):
            continue

        output = str(result.get("output", "")).strip()

        # ❌ Empty output → INVALID
        if output == "":
            continue

        validated.append({
            "input": str(raw_input),
            "expected": output
        })

    # ❌ If fewer than 2 valid cases → reject question
    if len(validated) < 2:
        raise HTTPException(
            status_code=422,
            detail="Reference solution could not validate enough test cases"
        )

    return validated[:3]  # cap for UI


def extract_projects_smart(resume_text: str) -> List[Dict[str, Any]]:
    """
    GENERALIZED project extraction – works for ANY resume format
    """
    projects = []
    lines = resume_text.split('\n')
    current = None

    PROJECT_INDICATORS = [
        r'\b(project|assignment|thesis|capstone)\b',
        r'\b(built|developed|created|implemented|designed|engineered)\b',
        r'\b(worked on|contributed to|led)\b'
    ]

    TECH_PATTERNS = {
        'languages': r'\b(python|java|javascript|c\+\+|go|rust|kotlin|swift|r)\b',
        'frameworks': r'\b(react|angular|vue|django|flask|spring|fastapi)\b',
        'databases': r'\b(sql|mysql|postgresql|mongodb|redis)\b',
        'cloud': r'\b(aws|azure|gcp|docker|kubernetes)\b',
        'ml_ai': r'\b(tensorflow|pytorch|scikit-learn|nlp|cnn|rnn)\b',
    }

    for line in lines:
        l = line.strip()
        ll = l.lower()

        if len(l) < 10:
            continue

        is_project = any(re.search(p, ll) for p in PROJECT_INDICATORS)

        if is_project:
            if current and current["description"]:
                projects.append(current)

            current = {
                "title": l[:150],
                "project_id": normalize_project_name(l),
                "description": "",
                "technologies": [],
                "domain": "general",
                "complexity_score": 0.0,
                "keywords": []
            }
        elif current:
            current["description"] += " " + l

    if current and current["description"]:
        projects.append(current)

    # Post processing
    for p in projects:
        desc = p["description"].lower()

        for _, pattern in TECH_PATTERNS.items():
            p["technologies"].extend(re.findall(pattern, desc, re.I))

        p["technologies"] = list(set(p["technologies"]))[:10]
        p["domain"] = classify_project_domain(p["description"])
        p["keywords"] = extract_action_keywords(p["description"])
        p["complexity_score"] = calculate_project_complexity(p)

    return sorted(projects, key=lambda x: x["complexity_score"], reverse=True)[:6]



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
            repair_resp = llm_call(repair_prompt, temperature=0.0, max_tokens=500)
        except Exception as e:
            repair_raws.append(f"exception:{e}")
            logger.warning("Repair llm_call exception attempt %d: %s", attempt, e)
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

import random

def build_generate_question_prompt(
    context: dict,
    mode: str = "first",
    required_type: str = "coding_challenge",
    state=None
) -> str:
    resume = context.get("resume", "")
    history = context.get("history", []) or []
    
    # 1. Build STRICT history context with Status Tags
    # This helps the LLM see if the previous topic was a failure
    recent_q_text_list = []
    for i, h in enumerate(history[-5:]):
        score = h.get("score", 1.0)
        status = "PASSED" if score >= 0.5 else "FAILED"
        recent_q_text_list.append(f"Q{i+1} [{status}]: {h.get('question','')[:120]}")
        
    recent_q_text = "\n".join(recent_q_text_list)
    
    diversity_hint = get_diverse_question_hint(history, required_type)
    round_context = ""
    if state:
        round_config = INTERVIEW_ROUNDS[state.current_round]
        round_progress = len(state.round_history[state.current_round]["questions"])
        
        round_context = f"""
═══════════════════════════════════════════════════════════════════
🎯 CURRENT ROUND: {round_config['name'].upper()}
Progress: Question {round_progress + 1} of {round_config['max_questions']}
Focus Areas: {', '.join(round_config['focus'])}
Pass Threshold: {round_config['pass_threshold'] * 100}%

⚠️ ELIMINATION ROUND: Candidate must maintain {round_config['pass_threshold'] * 100}%+ average to advance
═══════════════════════════════════════════════════════════════════
"""    
    difficulty_level = state.difficulty_level if state else "medium"

    # 2. Smart Project Selection
    projects = extract_projects_smart(resume)
    history_blob = " ".join([h.get("question", "").lower() for h in history])
    covered_projects = state.covered_projects if state else set()

    # Identify LAST project discussed to prevent immediate re-selection
    last_project_id = None
    if history:
        last_project_id = history[-1].get("target_project")

    available_projects = [
        p for p in projects 
        if p["project_id"] not in covered_projects 
        and p["project_id"] != last_project_id # <--- Added: Force switch if we just discussed this
        and p["title"].lower()[:15] not in history_blob
    ]
    
    target_project = None
    if required_type == "project_discussion":
        if available_projects:
            target_project = random.choice(available_projects)
        elif projects:
            # Force a different angle on a revisited project, but try to avoid the very last one
            candidates = [p for p in projects if p["project_id"] != last_project_id]
            if candidates:
                target_project = random.choice(candidates)
            else:
                target_project = random.choice(projects)
    
    # 3. Universal Strategy Controller with Resume Validation
    project_focus = "" 
    
    if required_type == "coding_challenge":
        # List already-asked algorithm types
        asked_algos = set()
        for h in history:
            q = h.get("question", "").lower()
            if "array" in q or "list" in q:
                asked_algos.add("arrays")
            if "string" in q or "substring" in q:
                asked_algos.add("strings")
            if "hash" in q or "map" in q or "dict" in q:
                asked_algos.add("hashmaps")
            if "tree" in q or "binary" in q:
                asked_algos.add("trees")
            if "graph" in q or "node" in q:
                asked_algos.add("graphs")
            if "sort" in q:
                asked_algos.add("sorting")
            if "search" in q:
                asked_algos.add("searching")
            if "dynamic" in q or "dp" in q:
                asked_algos.add("dynamic_programming")
        
        remaining_algos = ["arrays", "strings", "hashmaps", "trees", "graphs", "sorting", "searching", "dynamic programming", "two pointers", "sliding window"]
        remaining_algos = [a for a in remaining_algos if a not in asked_algos]
        
        project_focus = f"""
🎯 FOCUS: DATA STRUCTURES & ALGORITHMS (DSA)
- Ignore resume projects for this turn
- Algorithm types ALREADY ASKED: {', '.join(asked_algos) if asked_algos else 'None'}
- **MANDATORY**: Use one of these UNUSED types: {', '.join(remaining_algos[:5]) if remaining_algos else 'any algorithm'}
- Difficulty: {difficulty_level}
- Language: Python
- Create a solvable problem (not just theory)
- **CRITICAL**: This MUST be a coding problem with test cases, NOT a conceptual question
"""
    elif required_type == "system_design":
        project_focus = f"""
🎯 FOCUS: SYSTEM DESIGN (WHITEBOARD SESSION)
- The candidate has indicated backend/architecture skills.
- Ask a High-Level Design (HLD) question appropriate for a fresher/intern.
- **Classic Examples**: Design a URL Shortener, Design a Chat App, Design a Parking Lot, Design an Image Upload Service.
- **Constraints**:
  1. Ask for High Level Architecture (API, DB, Key Components).
  2. Do NOT ask for code. Ask for the *Structure*.
  3. Mention: "You can use the whiteboard to draw your components."
- Difficulty: {difficulty_level}
"""    
        
    elif required_type == "experience":
        # ✅ CHECK: Does resume have work experience?
        has_experience = any(kw in resume.lower() for kw in ['experience', 'work history', 'employment', 'intern at', 'developer at', 'engineer at', 'worked at'])
        
        if not has_experience:
            # Fallback to skills-based conceptual question
            project_focus = """
🎯 FOCUS: Core Technical Skills (No Work Experience Found)
- Scan the 'Skills' section for their STRONGEST technology
- Ask an advanced concept question about that skill
- Example: "Explain how garbage collection works in Java" or "What's the difference between REST and GraphQL?"
- Focus on trade-offs and real-world applications
"""
        else:
            project_focus = """
🎯 FOCUS: PROFESSIONAL EXPERIENCE (WORK HISTORY)
- SCAN the 'Experience' or 'Work History' section
- Identify a SPECIFIC company and role mentioned
- Ask about a SPECIFIC responsibility, achievement, or technical decision
- Example: "At [Company Name], you mentioned [specific task]. What was the technical challenge?"
- **FORBIDDEN**: Generic questions like "Tell me about your experience"
"""

    elif required_type == "achievement":
        # ✅ CHECK: Does resume have achievements?
        has_achievements = any(kw in resume.lower() for kw in ['achievement', 'award', 'competition', 'hackathon', 'certification', 'rank', 'winner', 'finalist', 'prize'])
        
        if not has_achievements:
            # Fallback to advanced skills question
            project_focus = """
🎯 FOCUS: Advanced Technical Concepts (No Achievements Found)
- Scan the 'Skills' section
- Ask about system design or architectural trade-offs
- Example: "When would you use NoSQL vs SQL databases?" or "How would you design a scalable API?"
- Focus on decision-making and real-world scenarios
"""
        else:
            project_focus = """
🎯 FOCUS: ACHIEVEMENTS, AWARDS & CERTIFICATIONS
- SCAN for competitions, hackathons, rankings, certifications
- Pick the MOST impressive entry
- Ask about the TECHNICAL challenge solved or skill acquired
- Example: "What algorithmic approach won you [Competition Name]?" or "How did you apply [Certification skill] in practice?"
"""
    elif required_type == "collaboration":
        project_focus = """
🎯 FOCUS: TEAMWORK & CONFLICT RESOLUTION (Behavioral)
- Ask about a time the candidate faced a disagreement or conflict in a team.
- Focus on: Communication, Empathy, and Resolution.
- Example: "Tell me about a time you disagreed with a teammate's technical decision. How did you resolve it?"
- **Constraint**: Do NOT ask about technical details, ask about the *interaction*.
"""

    # 🟢 NEW BLOCK 2: OWNERSHIP / LEADERSHIP
    elif required_type == "ownership":
        project_focus = """
🎯 FOCUS: OWNERSHIP & INITIATIVE (Behavioral)
- Ask about a time the candidate went above and beyond or took responsibility for a failure.
- Focus on: Accountability, Proactivity, and Learning.
- Example: "Tell me about a mistake you made in a project. How did you fix it and what did you learn?"
- **Constraint**: Look for "I" statements, not just "We".
"""

    # 🟢 NEW BLOCK 3: CATCH-ALL BEHAVIORAL (Safety Net)
    elif required_type == "behavioral" or required_type == "hr":
        project_focus = """
🎯 FOCUS: GENERAL BEHAVIORAL / CULTURE FIT
- Ask a standard behavioral question using the STAR method.
- Topics: Adaptability, Time Management, or Motivation.
- Example: "Describe a situation where you had to learn a new technology very quickly."
"""

    elif required_type == "project_discussion" and target_project:
        project_focus = f"""
🎯 FOCUS: PROJECT DEEP DIVE
- Target Project: "{target_project['title']}"
- Tech Stack: {', '.join(target_project.get('technologies', []))}
- **MANDATORY**: Ask about ONE of these aspects:
  • A specific bug they encountered and how they debugged it
  • Why they chose technology X over alternative Y
  • A performance optimization they made (with numbers/metrics)
  • A specific architectural decision (e.g., "Why microservices vs monolith?")
  • A challenging edge case they handled
  
- **FORBIDDEN**: Generic questions like "How did you build this?" or "What challenges did you face?"
- **REQUIRED**: Reference the exact project title in your question
"""
    
    else:
        project_focus = """
🎯 FOCUS: Advanced Technical Concepts
- Scan the 'Skills' section for their STRONGEST technology
- Ask about an ADVANCED concept in that technology
- Example: "Explain the event loop in Node.js" or "How does Python's GIL affect multithreading?"
- Focus on trade-offs, internals, or best practices
"""

    # 4. Build the final prompt with STRICT enforcement
    prompt = f"""
SYSTEM: You are a STRICT Technical Interviewer. Your goal is to VERIFY the candidate's claimed skills.

⚠️ CRITICAL RULES (VIOLATING THESE = FAILURE):
1. **NO REPETITION**: You MUST NOT ask about topics mentioned in previous questions
2. **BE SPECIFIC**: Quote exact claims from the resume (e.g., "You mentioned 'reduced latency by 40%'...")
3. **NO HALLUCINATIONS**: Only ask about content present in the resume below
4. **FORCE DEPTH**: Ask "HOW" and "WHY", not just "WHAT"
5. **TYPE ENFORCEMENT**: If type is coding_challenge, it MUST be a coding problem with test cases

═══════════════════════════════════════════════════════════════════
CANDIDATE RESUME:
{resume[:4000]}
{round_context}

═══════════════════════════════════════════════════════════════════
PREVIOUS QUESTIONS (DO NOT REPEAT THESE TOPICS):
{recent_q_text if recent_q_text else "None - This is the first question"}

{diversity_hint}

═══════════════════════════════════════════════════════════════════
CURRENT QUESTION TYPE: {required_type.upper().replace('_', ' ')}
DIFFICULTY: {difficulty_level.upper()}

{project_focus}

═══════════════════════════════════════════════════════════════════
OUTPUT FORMAT (JSON ONLY, NO MARKDOWN):
{{
  "question": "The interview question (must be SPECIFIC and DIFFERENT from previous)",
  "type": "{required_type}",
  "target_project": "{target_project['project_id'] if target_project else 'general'}",
  "difficulty": "{difficulty_level}",
  "coding_challenge": {{
      "language": "python",
     "starter_code": "def function_name(params):\n    # TODO: Implement this function\n    pass",
      "reference_solution": "def function_name(params):\\n    # FULL WORKING SOLUTION REQUIRED FOR VALIDATION\\n    return result",
      "test_cases": [
          {{"input": "\\"json_value\\"", "expected": "\\"expected_output\\""}},
          {{"input": "\\"different_input\\"", "expected": "\\"different_output\\""}}
      ]
  }}
}}
🚨 FINAL CHECK: Re-read the PREVIOUS QUESTIONS section. Is your new question about the SAME topic? If yes, CHANGE IT.
"""
    return prompt.strip()

# ==========================================
# SCORING SYSTEM
# ==========================================

def build_score_prompt(
    question_text: str, 
    ideal_outline: str, 
    candidate_answer: str, 
    context: Optional[Dict[str, Any]] = None,
    user_time_complexity: Optional[str] = None,
    user_space_complexity: Optional[str] = None
) -> str:
    """
    CAMPUS-LEVEL MULTI-DIMENSIONAL SCORING
    - Designed for Intern / Fresher / Placement interviews
    - Harsh on bluffing, fair on partial understanding
    - Supports TEXT, CODE, and SYSTEM DESIGN modes
    """
    context = context or {}
    resume = context.get("resume", "")
    chunks = context.get("chunks", [])
    
    question_type = context.get("question_type", "text")
    exec_result = context.get("code_execution_result", {})
    
    chunks_text = ""
    if chunks:
        chunks_text = "\n".join([
            f"[Ref {c.get('doc_id')}:{c.get('chunk_id')}] {c.get('snippet','')[:300]}"
            for c in chunks[:3]
        ])
    
    dimensions_text = "\n".join([
        f"- **{name}** (weight: {info['weight']}): {info['description']}"
        for name, info in SCORING_DIMENSIONS.items()
    ])

    # =========================================================
    # 1. CODE SCORING (Campus-Friendly)
    # =========================================================
    if question_type == "coding_challenge":
        passed = exec_result.get("passed", False)
        output_log = exec_result.get("output", "No output captured")
        error_type = exec_result.get("error", "None")
        complexity_check_block = ""
        if user_time_complexity or user_space_complexity:
            complexity_check_block = f"""
        📜 COMPLEXITY AUDIT (CRITICAL):
        The user CLAIMS:
        - Time: {user_time_complexity if user_time_complexity else "NOT PROVIDED"}
        - Space: {user_space_complexity if user_space_complexity else "NOT PROVIDED"}

        YOUR TASK:
        1. Analyze the 'CANDIDATE ANSWER' code yourself. Determine the ACTUAL Time/Space complexity.
        2. Compare ACTUAL vs CLAIMED.
        3. SCORING RULES:
           - If Claim is Missing: Deduct 0.10 from 'technical_accuracy'.
           - If Claim is WRONG (e.g. claimed O(n) but code is O(n^2)): Deduct 0.20 from 'technical_accuracy'. Mark as Weak.
           - If Claim is CORRECT: Award Bonus points.
        """ 
        system = f"""
You are a CAMPUS PLACEMENT CODE EVALUATOR.

CANDIDATE LEVEL:
- Student / Fresher / Intern
- Expect correct logic and clarity, not extreme optimization

EXECUTION STATUS:
- Passed: {passed}
- Error Type: {error_type}


EXECUTION LOG:
{output_log[:800]}

{complexity_check_block}
CODE SCORING RULES (IMPORTANT):
1. IF TESTS FAILED:
   - MAX SCORE = 0.5
   - Minor syntax / edge-case miss → 0.35-0.45
   - Logical misunderstanding → < 0.3

2. IF TESTS PASSED:
   - BASE SCORE = 0.7
   - +0.1 to +0.25 for:
     • clean logic
     • readable variable names
     • handling edge cases
     • correct use of data structures
     • CORRECT COMPLEXITY ANALYSIS (if provided)
   - Do NOT expect optimal Big-O unless resume claims it
3. **OPTIMIZATION & QUALITY (Multipliers)**:
   - **Brute Force vs Optimal**: If the problem can be solved in O(n) but user wrote O(n^2), score CANNOT exceed 0.75, even if tests pass.
   - **Code Style**: Penalize for single-letter variables (except i, j, n), lack of comments, or deep nesting.
   - **Edge Cases**: Did they handle empty inputs/nulls in the code logic?
4. STRICT CHEATING CHECK:
   - Hardcoded outputs → score = 0.0
   - Printing expected values → score = 0.0

SCORING DIMENSIONS:
{dimensions_text}

CAMPUS CONTEXT:
- Partial but honest attempts > perfect but copied solutions
- Penalize bluffing more than lack of knowledge
"""

    # =========================================================
    # 2. SYSTEM DESIGN SCORING (Whiteboard Consistency)
    # =========================================================
    elif question_type == "system_design":
        # Extract the text labels found in the diagram
        wb_summary = context.get("whiteboard_text_summary", "No whiteboard usage detected")
        
        system = f"""
You are a SENIOR SOFTWARE ARCHITECT evaluating a System Design interview.

CANDIDATE LEVEL: Intern / Entry Level
- Expect basic component knowledge (Load Balancer, DB, API, Cache).
- Do NOT expect complex distributed locking or CAP theorem nuance unless they bring it up.

INPUT DATA:
- **Verbal Explanation**: See 'CANDIDATE ANSWER' below.
- **Whiteboard Text Labels**: {wb_summary}

🕵️ WHITEBOARD LOGIC CHECK:
1. **Completeness**: Does the whiteboard list contain the 'Holy Trinity' of Web Apps?
   - Frontend/Client
   - Backend/API/Server
   - Database/Storage
   *If any of these 3 are missing from the text labels, CAP score at 0.6.*

2. **Specifics vs Generics**:
   - "Database" = Weak
   - "PostgreSQL" / "MongoDB" = Strong
   - "Cache" = Weak
   - "Redis" / "CDN" = Strong

3. **Scalability Indicators**:
   - Look for keywords: "Load Balancer", "Kafka", "Queue", "Sharding", "Replica".
   - If prompt asked for scaling and these are missing -> Deduct 'depth_of_understanding'.

SCORING CRITERIA:
- **0.0 - 0.4 (Fail)**: Whiteboard empty or completely disconnected from verbal answer.
- **0.5 - 0.6 (Weak)**: Generic boxes ("App", "DB") without specific technologies.
- **0.7 - 0.8 (Pass)**: Standard architecture present. Components connect logically.
- **0.9 - 1.0 (Strong)**: Specific tech choices (e.g., "S3 for images"), Scalability components (LBs, Queues), and clear data flow.
SCORING DIMENSIONS:
{dimensions_text}
OUTPUT GUIDANCE:
- If they drew nothing, mention "Candidate failed to utilize whiteboard" in the rationale.
"""
        
    # =========================================================
    # 3. TEXT SCORING (Standard / Fallback)
    # =========================================================
    else:
        system = f"""
You are a CAMPUS RECRUITER evaluating a technical answer for INTERNSHIP / PLACEMENT.

CANDIDATE CONTEXT:
- Student or fresher
- Answer should match what they claim in their resume
- Expect understanding, not expert depth

SCORING DIMENSIONS:
{dimensions_text}

SCORING GUIDELINES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FAIL (0.0-0.3):
• Completely incorrect or irrelevant
• Clearly bluffing about own resume project
• Uses buzzwords without understanding
• Cannot explain own implementation at all

WEAK (0.3-0.5):
• Generic explanation
• Knows terms but not flow
• Avoids specifics
• Sounds memorized

ACCEPTABLE (0.7-0.8):
• Understands core idea
• Explains basic implementation
• Minor gaps are OK
• Typical intern-level answer

STRONG (0.8-0.9):
• Explains HOW and WHY
• Mentions challenges faced
• Talks about trade-offs
• Clearly built it themselves

EXCEPTIONAL (0.9-1.0):
• Discusses debugging, edge cases, metrics
• Compares approaches
• Shows placement-ready confidence

CRITICAL BLUFF SIGNALS:
- Overuse of “we” instead of “I”
- Vague phrases: “industry best practices”
- No file names, functions, or examples
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CAMPUS EVALUATION RULE:
Honesty + reasoning > memorized perfection
"""

    # =========================================================
    # OUTPUT SCHEMA (UNCHANGED)
    # =========================================================
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
  "rationale": "string",
  "red_flags_detected": ["list"],
  "missing_elements": ["list"],
  "follow_up_probe": "string or null",
  "mentor_tip": "string"
}'''

    prompt = f"""SYSTEM:
{system}

QUESTION ASKED:
{question_text}

IDEAL ANSWER SHOULD COVER:
{ideal_outline}

CANDIDATE ANSWER:
{candidate_answer[:800]}

RESUME CONTEXT:
{resume[:600]}

REFERENCE MATERIAL:
{chunks_text}

INSTRUCTIONS:
1. Score using campus-level expectations
2. Penalize bluffing heavily
3. Be fair to partial but honest answers
4. Suggest a FOLLOW-UP only if score is 0.4-0.7
5. Provide a MENTOR TIP suitable for placement prep

OUTPUT JSON:
{schema}
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
        "stdin": stdin or ""# ms (20s)
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
    
    # 1. First check hard rules (Keep this! It handles catastrophic failure)
    hard_decision = check_termination_rules(context.get("question_history", []))
    
    if hard_decision:
        hard_decision["ended"] = True
        return {"ok": True, "parsed": hard_decision, "raw": "hard_rule_triggered"}

    # =========================================================================
    # 🛡️ SAFETY GUARD: PREVENT PREMATURE MODEL REJECTION (MISSING IN YOUR CODE)
    # =========================================================================
    history = context.get("question_history", [])
    # We need metrics to check the score
    metrics = calculate_performance_metrics(history) 
    
    # RULE: If interview is short (< 3 Qs) and score is okay (> 40%), FORCE CONTINUE.
    if len(history) < 3 and metrics["average_score"] > 0.40:
        logger.info(f"🛡️ Safety Guard Triggered: Forcing CONTINUE (Questions={len(history)}, Score={metrics['average_score']:.2f})")
        return {
            "ok": True, 
            "parsed": {
                "ended": False,
                "verdict": "maybe",
                "confidence": 1.0,
                "reason": "Interview is too short to make a final decision. Continuing to gather more signals."
            }, 
            "raw": "safety_guard_triggered"
        }
    # =========================================================================

    # 2. Otherwise, consult the model (Normal Flow)
    prompt = build_decision_prompt(context)
    resp = llm_call(prompt, temperature=temperature, max_tokens=600)
    
    if not resp.get("ok"):
        return {"ok": False, "parsed": None, "raw": resp.get("error")}
    
    parsed = extract_json_from_text(resp["raw"])
    if not parsed:
        # Fallback decision based on metrics
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
    resp = llm_call(prompt, temperature=0.0, max_tokens=500)
    
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
def build_full_context(parsed: dict) -> str:
    """
    Deterministically builds interview context.
    ZERO summarization.
    ZERO paraphrasing.
    """
    parts = []

    if parsed.get("skills"):
        parts.append("TECHNICAL SKILLS:\n" + ", ".join(parsed["skills"]))

    for p in parsed.get("projects", []):
        parts.append(
            f"\nPROJECT: {p.get('title','')}\n"
            f"Technologies: {', '.join(p.get('technologies', []))}\n"
            f"Description: {p.get('description','')}"
        )

    for w in parsed.get("work_experience", []):
        parts.append(
            f"\nWORK EXPERIENCE: {w.get('role','')} at {w.get('company','')}\n"
            f"{w.get('description','')}"
        )

    return "\n".join(parts)

def ai_parse_resume(text: str) -> dict:
    """
    Resume parsing using Groq 8B.
    LLM extracts STRUCTURE ONLY.
    We build full_context_for_prompt ourselves (NO summarization).
    """
    if not text or not text.strip():
        return regex_parse_resume(text)

    safe_text = text[:50000]

    prompt = f"""
    You are a Resume Information Extractor.

    Extract structured fields EXACTLY as written.
    Do NOT summarize.
    Do NOT paraphrase.
    Do NOT generalize.

    OUTPUT JSON ONLY:
    {{
        "name": "",
        "email": "",
        "phone": "",
        "skills": [],
        "education": [{{ "degree": "", "institution": "" }}],
        "projects": [
            {{
                "title": "",
                "technologies": [],
                "description": ""
            }}
        ],
        "work_experience": [
            {{
                "role": "",
                "company": "",
                "description": ""
            }}
        ]
    }}

    RESUME TEXT:
    {safe_text}
    """

    parsed = None

    # 1️⃣ Groq 8B
    if groq_client:
        try:
            resp = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=3000,
                response_format={"type": "json_object"}
            )
            raw = resp.choices[0].message.content
            parsed = extract_json_from_text(raw)
        except Exception as e:
            logger.warning(f"Groq parse failed: {e}")

    # 2️⃣ OpenRouter fallback
    if not parsed:
        try:
            resp = openrouter_client.chat.completions.create(
                model="meta-llama/llama-3.3-70b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=4000
            )
            parsed = extract_json_from_text(resp.choices[0].message.content)
        except Exception as e:
            logger.warning(f"OpenRouter fallback failed: {e}")

    if not parsed:
        return regex_parse_resume(text)

    # 🔥 BUILD FULL CONTEXT YOURSELF (NO AI)
    parsed["full_context_for_prompt"] = build_full_context(parsed)

    return parsed


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

QUESTION_TYPE_WEIGHTS = {
    "conceptual": 0.40,
    "project_discussion": 0.30,
    "coding_challenge": 0.30
}
def infer_type_from_question(q: str) -> str:
    q = q.lower()
    if any(x in q for x in ["write a function", "implement", "code", "algorithm"]):
        return "coding_challenge"
    if any(x in q for x in ["at ", "intern", "worked", "company"]):
        return "experience"
    if "project" in q:
        return "project_discussion"
    if any(x in q for x in ["ranked", "award", "competition", "finalist"]):
        return "achievement"
    return "conceptual"


class HintRequest(BaseModel):
    session_id: str
    question: str
    context_type: str = "conceptual"
    current_answer: Optional[str] = ""
class InterviewState:
    """
    Stateless Interview Manager.
    Re-calculates interview coverage from history on every request.

    Guarantees coverage of:
    - Projects
    - Experience (if present)
    - Achievements (if present)
    - DSA / Coding Challenges (minimum 2)
    """

    def __init__(self, resume_text: str):
        self.resume_text = resume_text

        # Extract projects once from resume
        self.projects = extract_projects_smart(resume_text)

        # Coverage trackers
        self.covered_projects: set[str] = set()
        self.history: List[Dict[str, Any]] = []
        self.visited_topics: set[str] = set()

        # Difficulty control
        self.difficulty_level = "medium"
        self.recent_scores: List[float] = []
        design_keywords = [
    'backend', 'architecture', 'scalable', 'microservices', 
    'distributed', 'api', 'database', 'system', 'design',
    'aws', 'cloud', 'docker', 'server', 'deployment','system_design'
]

        self.has_system_design_skills = any(kw in resume_text.lower() for kw in design_keywords)
        # Resume signals
        self.has_work_experience = any(
            kw in resume_text.lower()
            for kw in [
                "experience",
                "work history",
                "employment",
                "intern at",
                "developer at",
                "engineer at",
                "worked at",
            ]
        )
        self.has_achievements = any(
            kw in resume_text.lower() 
            for kw in ['achievement', 'award', 'competition', 'hackathon', 'certification', 'rank', 'winner', 'finalist', 'prize']
        )
        self.current_round = "screening"
        self.round_history = {
            "screening": {"questions": [], "scores": [], "status": "in_progress"},
            "technical": {"questions": [], "scores": [], "status": "not_started"},
            "behavioral": {"questions": [], "scores": [], "status": "not_started"}
        }
        self.eliminated = False
        self.elimination_reason = None

        # Section counters (hydrated every request)
        self.section_counts = {
            "experience": 0,
            "projects": 0,
            "achievements": 0,
            "dsa": 0,
            "conceptual": 0,
        }

    # =====================================================
    # 🔁 STATE HYDRATION (SOURCE OF TRUTH)
    # =====================================================
    def _update_round_status(self):
        """
        Checks round completion and elimination.
        Updates current_round pointer.
        """
        round_order = ["screening", "technical", "behavioral"]
        
        for round_name in round_order:
            round_data = self.round_history[round_name]
            round_config = INTERVIEW_ROUNDS[round_name]
            
            num_questions = len(round_data["questions"])
            scores = round_data["scores"]
            
            # Skip if not started
            if num_questions == 0:
                if round_data["status"] == "not_started":
                    self.current_round = round_name
                return
            
            # Check if round is complete (min questions met)
            if num_questions >= round_config["min_questions"]:
                avg_score = sum(scores) / len(scores) if scores else 0.0
                
                # ELIMINATION CHECK
                if avg_score < round_config["pass_threshold"]:
                    self.eliminated = True
                    self.elimination_reason = (
                        f"Failed {round_config['name']} "
                        f"(Score: {avg_score:.2f} < {round_config['pass_threshold']:.2f})"
                    )
                    round_data["status"] = "failed"
                    return
                
                # Round passed
                round_data["status"] = "passed"
                
                # Check if max questions reached (force advance)
                if num_questions >= round_config["max_questions"]:
                    continue  # Move to next round
                
                # If not at max, stay in current round
                self.current_round = round_name
                return
            else:
                # Still collecting minimum questions for this round
                self.current_round = round_name
                round_data["status"] = "in_progress"
                return
        
        # All rounds complete
        self.current_round = "complete"    
    def hydrate_from_history(self, history: List[Dict[str, Any]]):
        """
        Rebuilds state from history INCLUDING round tracking.
        """
        self.history = history or []
        self.covered_projects.clear()
        self.recent_scores.clear()
        
        # Reset round tracking
        for round_name in self.round_history:
            self.round_history[round_name] = {
                "questions": [], 
                "scores": [], 
                "status": "not_started"
            }
        
        self.current_round = "screening"
        self.eliminated = False
        
        scores: List[float] = []
        
        # Rebuild round state from history
        for h in self.history:
            q_type = h.get("type", "conceptual")
            
            # Skip probes from round counting (they don't advance rounds)
            if h.get("is_probe", False):
                continue
            
            # Determine which round this question belongs to
            round_name = QUESTION_TYPE_TO_ROUND.get(q_type, "screening")
            
            # Record question in round
            self.round_history[round_name]["questions"].append(h)
            
            # Record score if available
            s = h.get("score")
            if s is not None:
                try:
                    score_val = float(s)
                    self.round_history[round_name]["scores"].append(score_val)
                    scores.append(score_val)
                except:
                    pass
            
            # Track project coverage
            target = h.get("target_project")
            if isinstance(target, str) and target != "general":
                self.covered_projects.add(target)
        
        # Determine current round and check elimination
        self._update_round_status()
        
        # Adaptive difficulty
        self.recent_scores = scores
        if scores:
            avg = sum(scores[-3:]) / len(scores[-3:])
            if avg > 0.8:
                self.difficulty_level = "hard"
            elif avg < 0.4:
                self.difficulty_level = "easy"
            else:
                self.difficulty_level = "medium"

        # ---- adaptive difficulty ----
        self.recent_scores = scores
        if scores:
            avg = sum(scores[-3:]) / len(scores[-3:])
            if avg > 0.8:
                self.difficulty_level = "hard"
            elif avg < 0.4:
                self.difficulty_level = "easy"
            else:
                self.difficulty_level = "medium"

    # =====================================================
    # 🎯 CORE ROUTING LOGIC (GUARANTEED FLOW)
    # =====================================================
    def next_question_type(self) -> str:
        """
        Returns next question type based on current round.
        Probes are handled separately and don't affect round progression.
        """
        if self.eliminated:
            return "eliminated"
        
        if self.current_round == "complete":
            return "complete"
        
        round_config = INTERVIEW_ROUNDS[self.current_round]
        focus_areas = round_config["focus"]
        
        # Get questions already asked in current round (excluding probes)
        round_questions = self.round_history[self.current_round]["questions"]
        asked_types = [q.get("type") for q in round_questions if not q.get("is_probe", False)]
        valid_focus_areas = []
        for t in focus_areas:
            # Only allow system_design if resume supports it
            if t == "system_design" and not self.has_system_design_skills:
                continue
            # Only allow experience/achievement if applicable (existing logic)
            if t == "experience" and not self.has_work_experience:
                continue
            if t == "achievement" and not self.has_achievements:
                continue
            valid_focus_areas.append(t)
        if not valid_focus_areas:
            valid_focus_areas = ["conceptual", "coding_challenge"]

        # 2. Pick a type we haven't asked much yet
        # Try to ensure we ask at least 1 coding challenge in technical
        if self.current_round == "technical":
            if "coding_challenge" not in asked_types:
                return "coding_challenge"

        # General rotation logic
        available_types = [t for t in valid_focus_areas if asked_types.count(t) < 2]
        
        if available_types:
            # Prefer types not asked yet
            unused = [t for t in available_types if t not in asked_types]
            if unused:
                return unused[0]
            return available_types[0]
        
        # Fallback
        return valid_focus_areas[0]    
        # Try to pick diverse question type from focus areas
      
    # =====================================================
    # 🧩 HELPERS
    # =====================================================
    def record_question(self, *_args, **_kwargs):
        """
        Deprecated.
        State is rebuilt from history via hydrate_from_history.
        """
        pass

    def is_question_too_similar(self, new_question: str) -> bool:
        return is_repetitive_question(new_question, self.history)

INTERVIEW_STATE: Dict[str, InterviewState] = {}

    
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
    hint_used: Optional[bool] = False
    whiteboard_elements: Optional[List[Dict[str, Any]]] = None
    user_time_complexity: Optional[str] = None   # e.g. "O(n)"
    user_space_complexity: Optional[str] = None  # e.g. "O(1)"

# Update this class in main.py
class CodeSubmissionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")  # 🔥 CRITICAL

    language: str
    code: str
    stdin: Optional[Any] = ""                 # 🔥 must be Any
    expected_output: Optional[Any] = None     # 🔥 must be Any
    test_cases: Optional[List[Dict[str, Any]]] = []  # 🔥 Any, not str

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
        resp = llm_call(prompt, temperature=0.0, max_tokens=300)
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
def get_interview_state(session_id: str, resume: str) -> InterviewState:
    """
    Returns existing InterviewState for session
    or initializes it from resume text.
    """
    if session_id not in INTERVIEW_STATE:
        INTERVIEW_STATE[session_id] = InterviewState(resume)
    return INTERVIEW_STATE[session_id]
# --- REPLACE THE MAIN ENDPOINT ---

@app.post("/generate_question")
def generate_question(req: GenerateQuestionRequest):
    payload = req.dict()

    # =====================================================
    # 1. CONTEXT + STATE HYDRATION
    # =====================================================
    enforced = enforce_budget(payload)
    history = payload.get("question_history", []) or []
    enforced["history"] = history

    state = get_interview_state(
        payload["session_id"],
        payload.get("resume_summary", "")
    )
    state.hydrate_from_history(history)
    current_q_count = len([h for h in history if not h.get("is_probe", False)]) + 1
    if state.eliminated:
        return {
            "request_id": payload["request_id"],
            "q_count": current_q_count,
            "ended": True,
            "elimination": True,
            "reason": state.elimination_reason,
            "current_round": state.current_round,
            "round_history": state.round_history,
            "final_decision": {
                "verdict": "reject",
                "reason": state.elimination_reason,
                "confidence": 0.95
            }
        }
    rule_termination = check_termination_rules(history)
    if rule_termination:
        return {
            "request_id": payload["request_id"],
            "q_count": current_q_count,
            "ended": True,
            "elimination": True,
            "reason": rule_termination["reason"],
            "final_decision": rule_termination,
            "parsed": {"question": "Interview Terminated", "type": "info"}
        }
    if state.current_round == "complete":
        decision = call_decision({
            "resume": payload.get("resume_summary", ""),
            "question_history": history,
            "conversation": payload.get("conversation", []),
            "retrieved_chunks": enforced.get("chunks", [])
        }, temperature=0.0)

        return {
            "request_id": payload["request_id"],
            "q_count": current_q_count,
            "ended": True,
            "elimination": False,
            "current_round": "complete",
            "round_history": state.round_history,
            "final_decision": decision.get("parsed"),
        }
    required_type = state.next_question_type()
    logger.info(f"🎯 Round: {state.current_round} | Required type: {required_type}")


    # =====================================================
    # 🛑 TERMINATION CHECK (ONLY PLACE IT HAPPENS)
    # =====================================================
    if len(history) >= TERMINATION_RULES["max_questions"]:
        decision = call_decision({
            "resume": payload.get("resume_summary", ""),
            "question_history": history,
            "conversation": payload.get("conversation", []),
            "retrieved_chunks": enforced.get("chunks", [])
        }, temperature=0.0)

        return {
            "request_id": payload["request_id"],
            "q_count": current_q_count,
            "ended": True,
            "final_decision": decision.get("parsed"),
            "parsed": {"question": "Interview Complete", "type": "info"},
        }

    # =====================================================
    # 2. DETERMINE REQUIRED TYPE (ONCE)
    # =====================================================
    required_type = state.next_question_type()
    logger.info(f"🎯 Required question type: {required_type}")

    # =====================================================
    # 3. GENERATION LOOP
    # =====================================================
    MAX_RETRIES = 3
    parsed = None
    chosen_raw = None

    for attempt in range(MAX_RETRIES):
        temperature = 0.3 + attempt * 0.2

        prompt = build_generate_question_prompt(
            enforced,
            mode=payload.get("mode", "first"),
            required_type=required_type,
            state=state
        )

        if attempt > 0:
            prompt += "\n\n🚨 Previous attempt failed. Generate a DIFFERENT question."

        try:
            resp = llm_call(prompt, temperature=temperature, max_tokens=1200)
        except Exception:
            continue

        if not resp.get("ok"):
            continue

        raw = resp.get("raw", "")
        candidate = extract_json_from_text(raw)

        if not isinstance(candidate, dict):
            continue

        question = candidate.get("question")
        if not question:
            continue

        # ❌ Reject repetition
        if state.is_question_too_similar(question):
            continue

        # ❌ Reject topic overlap (except DSA)
        new_topics = extract_question_topics(question)
        old_topics = set().union(*[
            extract_question_topics(h.get("question", ""))
            for h in state.history
        ])

        if (
            required_type != "coding_challenge"
            and len(new_topics & old_topics) / max(len(new_topics), 1) > 0.6
        ):
            continue

        # 🔒 HARD TYPE ENFORCEMENT (NO SILENT SKIP)
        candidate["type"] = required_type

        # 🔒 Coding challenge enforcement
# 🔒 Coding challenge enforcement
        if required_type == "coding_challenge":
            cc = candidate.get("coding_challenge", {})
            tcs = cc.get("test_cases", [])
            
            # 1. Structural Check (Keep this to ensure frontend doesn't crash)
            if (
                not isinstance(tcs, list) 
                or len(tcs) < 2 
                or any("input" not in tc or "expected" not in tc for tc in tcs)
            ):
                try:
                    enforce_test_cases_for_challenge(
                        parsed=candidate, 
                        resp_raw=raw, 
                        original_prompt=prompt
                    )
                    cc = candidate["coding_challenge"]  # 🔥 re-fetch after repair
                    tcs = cc.get("test_cases", [])
                except HTTPException:
                    logger.warning("Failed to repair structure. Retrying...")
                    continue  # reject question
            
            # 2. ⚠️ VALIDATION REMOVED: Trust the LLM directly
            # We are NOT running the reference solution anymore. 
            # This prevents the "failing on validation" error.
            
            cc["test_cases"] = tcs[:3]

            # Security: Remove the solution so the user doesn't see it
            if "reference_solution" in cc:
                del cc["reference_solution"]
        
        # ✅ SUCCESS: Accept the candidate
        parsed = candidate
        chosen_raw = raw
        break
    # =====================================================
    # 4. FALLBACK (RARE, SAFE)
    # =====================================================
    if parsed is None:
        parsed = FALLBACK_QUESTIONS["conceptual"].copy()
        parsed["type"] = "conceptual"
        parsed["_is_fallback"] = True
        chosen_raw = "FALLBACK_TRIGGERED"

    # =====================================================
    # 5. FINAL NORMALIZATION (NO STATE MUTATION)
    # =====================================================
    parsed.setdefault("domain", "general")
    parsed.setdefault("target_project", "general")
    parsed["difficulty"] = state.difficulty_level

    # Ensure coding fields
    if parsed["type"] == "coding_challenge":
        cc = parsed.get("coding_challenge", {})
        tcs = cc.get("test_cases", [])
        
        # If test cases are missing/empty, FORCE generation right now
        if not tcs or len(tcs) < 2:
            logger.warning(f"⚠️ Coding challenge missing test cases. repairing: {question[:50]}...")
            q_text = parsed.get("question", "")
            # Use a specialized prompt to get just the test cases
            starter = cc.get("starter_code", "def solve(x):\n    pass")
            repaired_tcs = generate_missing_test_cases(q_text, starter)
            try:
                repaired_tcs = repaired_tcs[:3]
    
            except HTTPException:
                parsed = FALLBACK_QUESTIONS["coding_challenge"].copy()

                parsed["_is_fallback"] = True

            if repaired_tcs and len(repaired_tcs) >= 2:
                cc["test_cases"] = repaired_tcs
                # Fix legacy fields for frontend compatibility
                cc["test_case_input"] = repaired_tcs[0]["input"]
                cc["expected_output"] = repaired_tcs[0]["expected"]
                parsed["coding_challenge"] = cc
                logger.info(f"✅ Auto-repaired {len(repaired_tcs)} test cases.")
            else:
                # If repair fails, DOWNGRADE to conceptual so the app doesn't crash
                logger.warning("❌ Repair failed. Downgrading question to 'conceptual'.")
                parsed = FALLBACK_QUESTIONS["coding_challenge"].copy()

                # Remove the broken coding_challenge object
                parsed["_is_fallback"] = True

    if parsed["type"] == "coding_challenge":
        cc = parsed.setdefault("coding_challenge", {})
        cc.setdefault("language", "python")
        cc.setdefault("starter_code", "def solve(x):\n    pass")
        

        if not parsed.get("_is_fallback"):
            tcs = cc.get("test_cases", [])
            if tcs:
                cc["test_case_input"] = tcs[0]["input"]
                cc["expected_output"] = tcs[0]["expected"]

    # =====================================================
    # 6. RETURN (STATE IS DERIVED NEXT REQUEST)
    # =====================================================
    return {
        "request_id": payload["request_id"],
        "q_count": current_q_count,
        "parsed": parsed,
        "llm_raw": chosen_raw,
        "metadata": {
            "required_type": parsed["type"],
            "difficulty": state.difficulty_level,
            "covered_projects": list(state.covered_projects),
                 "current_round": state.current_round,  # NEW
            "round_progress": {  # NEW
                "screening": len(state.round_history["screening"]["questions"]),
                "technical": len(state.round_history["technical"]["questions"]),
                "behavioral": len(state.round_history["behavioral"]["questions"])
            }
        },
        "ended": False
    }


@app.post("/run_code")
def run_code(req: CodeSubmissionRequest):
    import json, re

    # ---------- 1. NORMALIZE STDIN ----------
    if isinstance(req.stdin, (list, dict)):
        req.stdin = json.dumps(req.stdin)
    elif req.stdin is None:
        req.stdin = ""

    # ---------- 2. NORMALIZE TEST CASES ----------
    clean_test_cases = []
    for tc in req.test_cases or []:
        inp = tc.get("input") or tc.get("stdin") or tc.get("test_case_input") or ""
        exp = tc.get("expected") or tc.get("expected_output") or tc.get("output") or ""
        clean_test_cases.append({"input": str(inp), "expected": str(exp)})

    final_code = req.code

    # ---------- 3. DRIVER INJECTION (PYTHON ONLY) ----------
    if (
        req.language.lower() == "python"
        and "def " in final_code
        and "if __name__" not in final_code
    ):
        target_func = "solve"
        match = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", final_code)
        if match:
            target_func = match.group(1)

        driver = f'''
import sys, json, inspect, traceback

def _parse_input(raw):
    raw = raw.strip()
    if raw == "":
        return None
    try:
        return json.loads(raw)
    except:
        return raw

if __name__ == "__main__":
    try:
        raw_input = sys.stdin.read()
        input_data = _parse_input(raw_input)

        sig = inspect.signature({target_func})
        params = list(sig.parameters)

        # ---------- STRICT & DETERMINISTIC DISPATCH ----------
        if len(params) == 0:
            result = {target_func}()
        elif len(params) == 1:
            result = {target_func}(input_data)
        else:
            if not isinstance(input_data, list):
                raise ValueError("Expected list input for multiple parameters")
            result = {target_func}(*input_data)

        if result is None:
            print("null")
        else:
            try:
                print(json.dumps(result))
            except:
                print(json.dumps(str(result)))

    except Exception as e:
        print("DRIVER_ERROR")
        print(json.dumps({{"error": str(e), "traceback": traceback.format_exc()}}))
'''
        final_code += "\n\n" + driver

    # ---------- 4. SELECT TEST CASES ----------
    cases_to_run = clean_test_cases or [{
        "input": req.stdin or "",
        "expected": req.expected_output or ""
    }]

    # ---------- 5. EXECUTION LOOP ----------
    results = []
    all_passed = True

    def normalize(val):
        if val is None:
            return None
        if isinstance(val, str):
            v = val.strip()
            try:
                return json.loads(v)
            except:
                pass
            try:
                return json.loads(f"[{v}]")
            except:
                pass
            return v
        return val

    for case in cases_to_run:
        c_input = case["input"]
        c_expected = case["expected"]

        run_result = run_code_in_sandbox(req.language, final_code, c_input)

        stdout_raw = run_result.get("output", "")
        stdout = str(stdout_raw).strip()

        passed = False
        error_msg = None

        if "DRIVER_ERROR" in stdout:
            error_msg = stdout
        else:
            if run_result.get("success") and c_expected != "":
                norm_out = normalize(stdout)
                norm_exp = normalize(c_expected)
                passed = (norm_out == norm_exp) or (str(norm_out) == str(norm_exp))

        if not passed:
            all_passed = False

        results.append({
            "input": c_input,
            "expected": c_expected,
            "stdout": stdout,
            "passed": passed,
            "success": run_result.get("success", False),
            "error": run_result.get("error_type") or error_msg
        })

    return {
        "success": True,
        "all_passed": all_passed,
        "results": results
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
@app.post("/generate_hint")
def generate_hint(req: HintRequest):
    """
    Context-Aware Socratic Hint. 
    Analyzes the user's partial answer to give specific, unblocking advice.
    """
    q_type = req.context_type
    partial = req.current_answer.strip() if req.current_answer else ""
    partial = safe_truncate(partial, 1500) # prevent context overflow

    # 1. Define Persona & Strategy
    if q_type == "coding_challenge":
        system_instruction = """
        You are a Senior Engineer mentoring a student.
        The candidate is writing code but is stuck.
        
        ANALYSIS STRATEGY:
        1. If 'PARTIAL CODE' is empty: Suggest a high-level approach or Data Structure (e.g., "Try a Hash Map").
        2. If 'PARTIAL CODE' has syntax errors: Point them out gently.
        3. If 'PARTIAL CODE' has bad logic (e.g., O(n^2)): Hint at optimization.
        4. If they are almost done: Suggest checking edge cases.
        
        DO NOT write the corrected code. Nudge them.
        """
    elif q_type == "system_design":
        system_instruction = """
        You are a System Architect.
        The candidate is designing a system on a whiteboard.
        
        ANALYSIS STRATEGY:
        1. If empty: Suggest starting with Functional Requirements or API definition.
        2. If they have components but no DB: "How will you store the data?"
        3. If they have a DB but no Scale: "How will this handle 1M users?"
        """
    else:
        system_instruction = """
        You are an Interviewer.
        The candidate is answering a conceptual question.
        If they are off-track, guide them back. If they are stuck, give an analogy.
        """

    prompt = f"""
    SYSTEM: {system_instruction}
    
    INTERVIEW QUESTION:
    "{req.question}"
    
    CANDIDATE'S CURRENT PARTIAL WORK:
    ```
    {partial if partial else "(Candidate has not started yet)"}
    ```
    
    OUTPUT:
    Provide a single, specific hint (max 2 short sentences).
    """
    
    try:
        # Temperature 0.3 for helpful, consistent advice
        resp = llm_call(prompt, temperature=0.3, max_tokens=150)
        
        if not resp.get("ok"):
             raise HTTPException(status_code=502, detail="AI hint generation failed")
        
        raw_hint = resp["raw"].strip().replace('"', '')
        return {"hint": raw_hint}
    except Exception as e:
        logger.error(f"Hint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))    
@app.post("/score_answer")
def score_answer(req: ScoreAnswerRequest):
    """
    Score answer using Groq (Llama 3.3) for ultra-low latency.
    Falls back to OpenRouter only if Groq fails.
    """
    payload = req.dict()
    
    # 1. PII redaction
    redaction_log = []
    if not payload.get("allow_pii") and payload.get("resume_summary"):
        r = redact_pii(payload["resume_summary"])
        payload["resume_summary"] = r["redacted"]
        redaction_log = r["redaction_log"]
    
    enforced = enforce_budget(payload)
    whiteboard_context = ""
    if payload.get("question_type") == "system_design":
        elements = payload.get("whiteboard_elements", [])
        whiteboard_context = extract_whiteboard_keywords(elements)
        logger.info(f"🎨 Whiteboard Analysis: {whiteboard_context}")
    # 2. Build Context & Prompt
    context = {
        "resume": enforced.get("resume", ""),
        "chunks": enforced.get("chunks", []),
        "question_type": payload.get("question_type", "text"),
        "code_execution_result": payload.get("code_execution_result"),
        "whiteboard_text_summary": whiteboard_context
    }
    
    prompt = build_score_prompt(
        payload.get("question_text", ""),
        payload.get("ideal_outline", ""),
        payload.get("candidate_answer", ""),
        context=context,
        user_time_complexity=payload.get("user_time_complexity"),
        user_space_complexity=payload.get("user_space_complexity")
    )
    
    # 3. FAST PATH: Try Groq First (Llama 3.3)
    parsed = None
    used_source = "groq"
    
    if groq_client:
        try:
            resp = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1500,
                response_format={"type": "json_object"}
            )
            raw_text = resp.choices[0].message.content
            parsed = extract_json_from_text(raw_text)
        except Exception as e:
            logger.warning(f"Groq Scoring failed: {e}")
            used_source = "openrouter_fallback"

    # 4. SLOW PATH: Fallback to OpenRouter if Groq failed/missing
    if not parsed:
        resp = llm_call(prompt, temperature=0.1, max_tokens=1500)
        if not resp.get("ok"):
             raise HTTPException(status_code=502, detail=f"Scoring failed: {resp.get('error')}")
        parsed = extract_json_from_text(resp["raw"])
        raw_text = resp["raw"]

    # 5. Validation & Post-Processing
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
            
            # Meta fields
            validated["confidence"] = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
            validated["verdict"] = parsed.get("verdict", "weak")
            validated["rationale"] = parsed.get("rationale", "")
            validated["red_flags_detected"] = parsed.get("red_flags_detected", [])
            validated["missing_elements"] = parsed.get("missing_elements", [])
            validated["follow_up_probe"] = parsed.get("follow_up_probe")
            
            # Stabilize Score
            validated["overall_score"] = normalize_overall_score(
                  validated, 
                  payload.get("question_history", [])
            )
            if payload.get("hint_used"):
                original_score = validated["overall_score"]
                penalized_score = original_score*0.85
                validated["overall_score"] = round(penalized_score,2)
                validated["rationale"] += " (Score reduced by 15% due to hint usage.)"
                logger.info(f"💡 Hint Penalty Applied: {original_score} -> {validated['overall_score']}")
            validated["verdict"] = derive_verdict_from_score(validated["overall_score"])
            
            # Record state
            state = INTERVIEW_STATE.get(payload["session_id"])
            if state:
                state.record_question(
                    {
                        "question": payload.get("question_text"),
                        "type": payload.get("question_type", "text"),
                        "target_project": payload.get("target_project", "general"),
                        "domain": payload.get("domain", "general"),
                    },
                    validated.get("overall_score", 0.5)
                )

            # Gray zone check (Initial Calculation)
            if validated["overall_score"] is not None:
                rules = TERMINATION_RULES
                if rules["gray_zone_min"] <= validated["overall_score"] <= rules["gray_zone_max"]:
                    needs_review = True
                    if not validated["follow_up_probe"]:
                        validated["follow_up_probe"] = "Ask for specific code example or implementation detail"
            
            if validated["confidence"] < 0.4:
                needs_review = True
            
        except Exception as e:
            logger.exception(f"Score validation failed: {e}")
            needs_review = True
    else:
        needs_review = True

    incoming_history = payload.get("question_history", []) or []
    current_q_count = len(incoming_history) + 1

    # ========================================================================
    # 🛑 ENHANCED ANTI-LOOP MECHANISM (MAX 1 PROBE PER TOPIC)
    # ========================================================================
    
    # 1. Calculate raw Gray Zone status
    raw_in_gray_zone = (
        needs_review 
        and validated["overall_score"] is not None 
        and TERMINATION_RULES["gray_zone_min"] <= validated["overall_score"] <= TERMINATION_RULES["gray_zone_max"]
    )
    
    # 2. STRICT PROBE SUPPRESSION LOGIC
    final_in_gray_zone = False  # Default: NO PROBE
    
    if raw_in_gray_zone and incoming_history:
        # ====================================================================
        # RULE 1: Check if we JUST asked a similar question (Probe Detection)
        # ====================================================================
        if len(incoming_history) >= 2:
            last_q = incoming_history[-1].get("question", "").lower()
            prev_q = incoming_history[-2].get("question", "").lower()
            
            similarity = compute_similarity(last_q, prev_q)
            
            # If questions are >25% similar, we JUST probed. STOP.
            if similarity > 0.25:
                logger.info(f"🛑 Anti-Loop Rule 1: Question similarity {similarity:.2f} > 0.25. NO PROBE.")
                final_in_gray_zone = False
            
            # ================================================================
            # RULE 2: Check Topic Overlap (Prevent drilling same concept)
            # ================================================================
            elif similarity <= 0.25:
                # Extract technical topics from both questions
                last_topics = extract_question_topics(last_q)
                prev_topics = extract_question_topics(prev_q)
                
                # If >50% topic overlap, we're beating a dead horse
                if last_topics and prev_topics:
                    overlap_ratio = len(last_topics & prev_topics) / max(len(last_topics), 1)
                    if overlap_ratio > 0.5:
                        logger.info(f"🛑 Anti-Loop Rule 2: Topic overlap {overlap_ratio:.2f} > 0.5. NO PROBE.")
                        final_in_gray_zone = False
                    else:
                        # ONLY allow probe if BOTH similarity AND topic overlap are low
                        final_in_gray_zone = True
                        logger.info(f"✅ Probe allowed: similarity={similarity:.2f}, topic_overlap={overlap_ratio:.2f}")
                else:
                    # If we can't extract topics, err on the side of NO PROBE
                    final_in_gray_zone = False
        
        # ====================================================================
        # RULE 3: Check if Previous Answer was ALSO Weak (Consecutive Weakness)
        # ====================================================================
        if final_in_gray_zone:  # Only check if not already suppressed
            prev_score = incoming_history[-1].get("score")
            if prev_score is not None:
                try:
                    prev_val = float(prev_score)
                    # If last question was ALSO in gray zone, candidate is stuck. Move on.
                    if TERMINATION_RULES["gray_zone_min"] <= prev_val <= TERMINATION_RULES["gray_zone_max"]:
                        logger.info(f"🛑 Anti-Loop Rule 3: Previous score {prev_val:.2f} was also gray. NO PROBE.")
                        final_in_gray_zone = False
                except Exception:
                    pass
        
        # ====================================================================
        # RULE 4: Hard Limit - Never Probe After Question 5
        # ====================================================================
        if final_in_gray_zone and current_q_count > 5:
            logger.info(f"🛑 Anti-Loop Rule 4: Question #{current_q_count} > 5. NO MORE PROBES.")
            final_in_gray_zone = False
    
    # ========================================================================
    # 3. DEBUGGING LOG (Shows why decision was made)
    # ========================================================================
    logger.info(
        f"Probe Decision | Q#{current_q_count} | Score: {validated['overall_score']:.2f} | "
        f"Raw Gray Zone: {raw_in_gray_zone} | Final Probe: {final_in_gray_zone}"
    )
    is_probe = False
    if len(incoming_history) >= 1:
        last_q = incoming_history[-1].get("question", "").lower()
        current_q = payload.get("question_text", "").lower()
        similarity = compute_similarity(last_q, current_q)
        if similarity > 0.25:
            is_probe = True    
    # ========================================================================
    # END ANTI-LOOP MECHANISM
    # ========================================================================

    return {
        "request_id": payload["request_id"],
        "q_count": current_q_count,
        "llm_raw": raw_text,
        "parsed": parsed,
        "validated": validated,
        "parse_ok": parsed is not None,
        "needs_human_review": needs_review,
        "source": used_source,
        "in_gray_zone": final_in_gray_zone,
        "is_probe": is_probe,  # Uses the suppressed value
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
    
    # ✅ Calculate metrics FIRST (fixes bug)
    metrics = calculate_performance_metrics(payload.get("question_history", []))
    
    context = {
        "resume": enforced.get("resume", ""),
        "conversation": payload.get("conversation", []),
        "question_history": payload.get("question_history", []),
        "retrieved_chunks": enforced.get("chunks", [])
    }
    
    result = call_decision(context, temperature=0.0)
    
    # -------------------------------------------------
    # FINALITY LOGIC (RULE-FIRST, MODEL-SECOND)
    # -------------------------------------------------
    is_final = False
    
    if result.get("ok") and result.get("parsed"):
        decision = result["parsed"]
        verdict = decision.get("verdict")
        confidence = decision.get("confidence", 0.0)

        # 🔒 HARD RULES ALWAYS FINAL
        if result.get("raw") == "hard_rule_triggered":
            is_final = True

        # 🔒 CLEAR MODEL DECISION
        elif payload.get("accept_model_final", True):
            if verdict in ("hire", "reject") and confidence >= 0.75:
                is_final = True

            # 🔒 LOW AVERAGE + REJECT = FINAL
            elif verdict == "reject" and metrics["average_score"] < 0.45:
                is_final = True

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
        parsed = ai_parse_resume(raw_text)
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
@app.get("/coverage/{session_id}")
def get_coverage(session_id: str):
    """Debug endpoint to see interview coverage"""
    if session_id not in INTERVIEW_STATE:
        return {"error": "Session not found"}
    
    state = INTERVIEW_STATE[session_id]
    
    uncovered = [
        p["title"] for p in state.projects 
        if p["project_id"] not in state.covered_projects
    ]
    
    covered = [
        p["title"] for p in state.projects 
        if p["project_id"] in state.covered_projects
    ]
    
    return {
        "session_id": session_id,
        "total_questions": len(state.history),
        "difficulty": state.difficulty_level,
        "section_counts": state.section_counts,
        "projects": {
            "total": len(state.projects),
            "covered": len(covered),
            "covered_list": covered,
            "uncovered_list": uncovered
        },
        "visited_topics": list(state.visited_topics)[:20]  # Top 20 topics
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
    - Returns 200 OK for violations (so frontend handles them as valid checks).
    - Returns 400 only for session/image errors.
    """
    # 1. Validate Session
    if req.session_id not in FACE_DB:
        # This is a setup error, so 400 is appropriate here
        return JSONResponse(status_code=400, content={"verified": False, "error": "Session not found."})

    reference_embedding = FACE_DB[req.session_id]["embedding"]

    # 2. Decode image
    img2 = decode_base64_image(req.current_image)
    if img2 is None:
        return JSONResponse(status_code=400, content={"verified": False, "error": "Image decode failed"})

    # =========================================================================
    # CHECK 1: COUNT FACES with MTCNN
    # =========================================================================
    try:
        face_objs = DeepFace.extract_faces(
            img_path=img2,
            detector_backend="mtcnn",   # Strict detector
            enforce_detection=False,    # Don't crash if 0 faces
            align=True
        )
        
        # Filter low confidence (ghost faces)
        valid_faces = [f for f in face_objs if f.get('confidence', 0) > 0.80]
        face_count = len(valid_faces)
        
    except Exception as e:
        logger.error(f"MTCNN error: {e}")
        face_count = 0

    # =========================================================================
    # CHECK 2: PROHIBITED OBJECTS (YOLO Only)
    # =========================================================================
    detected_items = []
    try:
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
    # DECISION LOGIC (UPDATED: Return Dicts = 200 OK)
    # =========================================================================
    
    # 1. Check Multiple People
    if face_count > 1:
        return {
            "verified": False,
            "violation_type": "multiple_people",
            "error": "Multiple people detected",
            "person_count": face_count,
            "details": f"MTCNN detected {face_count} distinct faces."
        }
    
    # 2. Check Objects
    if detected_items:
        return {
            "verified": False, 
            "violation_type": "prohibited_object", 
            "objects": detected_items, 
            "error": "Prohibited object detected"
        }

    # 3. Check No Face
    if face_count == 0:
         return {
             "verified": False, 
             "error": "No face detected", 
             "violation_type": "no_face_detected"
         }

    # =========================================================================
    # CHECK 3: IDENTITY MATCH (VGG-Face via MTCNN)
    # =========================================================================
    try:
        embedding_objs = DeepFace.represent(
            img_path=img2,
            model_name="VGG-Face",
            detector_backend="mtcnn",
            enforce_detection=True
        )
        
        current_embedding = embedding_objs[0]["embedding"]
        
        distance = cosine(reference_embedding, current_embedding)
        
        if distance <= STRICT_DISTANCE_THRESHOLD:
            return {"verified": True, "distance": distance}
        else:
            # Return 200 OK with verified=False
            return {
                "verified": False, 
                "distance": distance, 
                "error": "Face mismatch", 
                "violation_type": "face_mismatch"
            }

    except Exception as e:
        # Return 200 with error so frontend doesn't crash
        return {
            "verified": False, 
            "error": f"Identity check failed: {str(e)}"
        }
def health():
    return {"status": "healthy", "model": GROQ_MODEL}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)