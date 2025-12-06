# ai-interview-ai-service/main.py
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional, Dict, Any
from groq import Groq
from dotenv import load_dotenv
import os
import io
import json
import re
import tempfile

# text extraction libraries
import pdfplumber
import docx2txt

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set in environment (.env)")

# Allow model override via env for flexibility
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

groq_client = Groq(api_key=GROQ_API_KEY)

app = FastAPI(title="AI Interview Service (Groq Parser)")

# --- Models ------------------------------------------------
class EvalRequest(BaseModel):
    question_id: Optional[str] = None
    question_text: str
    ideal_outline: Optional[str] = ""
    candidate_answer: str
    resume_context: Optional[dict] = {}

# --- simple fallback (rule-based) utilities -----------------
SKILL_CANDIDATES = [
    "python","java","javascript","typescript","react","node","fastapi","flask","django","sql","postgresql",
    "mysql","mongodb","docker","kubernetes","aws","gcp","azure","tensorflow","pytorch","scikit-learn",
    "nlp","computer vision","react native","html","css","c++","c#","go","rust","git","ci/cd","graphql",
]

EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
PHONE_RE = re.compile(r"(\+?\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}|\d{10})")
YEARS_RE = re.compile(r"(?:\b(19|20)\d{2}\b)")
EDU_KEYWORDS = ["bachelor", "master", "phd", "b\\.tech", "m\\.tech", "btech", "msc", "bsc", "degree", "diploma"]

def extract_text_from_pdf_bytes(b: bytes) -> str:
    text_parts = []
    with pdfplumber.open(io.BytesIO(b)) as pdf:
        for page in pdf.pages:
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            if txt:
                text_parts.append(txt)
    return "\n".join(text_parts)

def extract_text_from_docx_bytes(b: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(b)
        tmp_path = tmp.name
    try:
        text = docx2txt.process(tmp_path) or ""
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
    return text

def extract_text_from_path(path: str) -> str:
    if not os.path.exists(path):
        return ""
    lower = path.lower()
    try:
        if lower.endswith(".pdf"):
            with open(path, "rb") as f:
                return extract_text_from_pdf_bytes(f.read())
        elif lower.endswith(".docx"):
            with open(path, "rb") as f:
                return extract_text_from_docx_bytes(f.read())
        else:
            with open(path, "r", errors="ignore") as f:
                return f.read()
    except Exception:
        return ""

def simple_skill_match(text: str):
    text_l = text.lower()
    found = []
    for s in SKILL_CANDIDATES:
        if re.search(r"\b" + re.escape(s) + r"\b", text_l):
            found.append(s)
    return sorted(set(found), key=lambda x: text_l.find(x))

def estimate_years_experience(text: str):
    years = [int(m.group(0)) for m in YEARS_RE.finditer(text)]
    years = [y for y in years if 1900 <= y <= 2100]
    if len(years) >= 2:
        return max(years) - min(years)
    m = re.search(r"(\d+)\+?\s+(?:years|yrs)\s+of\s+experience", text.lower())
    if m:
        return int(m.group(1))
    return None

def extract_education_lines(text: str):
    lines = []
    for line in text.splitlines():
        ll = line.strip().lower()
        if any(k in ll for k in EDU_KEYWORDS):
            lines.append(line.strip())
    return lines[:6]

def extract_name_candidate(text: str):
    for line in text.splitlines():
        l = line.strip()
        if not l:
            continue
        if EMAIL_RE.search(l) or PHONE_RE.search(l):
            continue
        if len(l.split()) <= 6 and len(l) > 2 and re.search(r"[A-Za-z]", l):
            return l.strip()
    return None

def extract_contact(text: str):
    email = EMAIL_RE.search(text)
    phone = PHONE_RE.search(text)
    return {
        "email": email.group(0) if email else None,
        "phone": phone.group(0) if phone else None
    }

def build_summary(text: str, max_chars=800):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if paragraphs:
        summary = " ".join(paragraphs[:2])
        return summary[:max_chars]
    return (text[:max_chars] if text else "")

# --- Groq (LLM) helper --------------------------------------
def extract_json_from_text(s: str) -> Optional[dict]:
    """
    Attempt to find a JSON object in the model output and parse it.
    """
    idx = s.find("{")
    if idx == -1:
        return None
    try:
        j = s[idx:]
        for end in range(len(j), 0, -1):
            try:
                candidate = j[:end]
                return json.loads(candidate)
            except Exception:
                continue
    except Exception:
        return None
    return None

def groq_parse_resume(text: str, max_retries: int = 2) -> dict:
    """
    Parse resume via Groq; if Groq doesn't return a summary, synthesize a concise one.
    Always returns 'summary' as a short string (or 'No summary available' if absolutely nothing).
    """
    model_name = os.getenv("GROQ_MODEL", GROQ_MODEL)

    prompt = f"""
You are a highly accurate resume parser. Extract structured information from the resume text below.

Return ONLY valid JSON (no surrounding markdown or extra commentary) with these fields:
{{ 
  "name": null,
  "email": null,
  "phone": null,
  "skills": [],
  "experience_years": null,
  "education": [],
  "summary": ""
}}

IMPORTANT:
- The "summary" field must be a short professional summary of the candidate in plain text, max 240 characters.
- If the resume contains no explicit summary, create a concise 1-2 sentence professional summary using the candidate's most relevant skills, degree, and years of experience if present.
Respond with strict JSON only.

Resume text:
\"\"\" 
{text}
\"\"\" 
"""
    for attempt in range(max_retries + 1):
        try:
            resp = groq_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1500,
            )

            # SDK shape handling
            content = ""
            if hasattr(resp, "choices") and len(resp.choices) > 0:
                content = getattr(resp.choices[0].message, "content", "") or resp.choices[0].message.content
            else:
                content = str(resp)

            print(f"GROQ raw content (attempt {attempt}): {content[:1200]}")

            parsed = extract_json_from_text(content)
            if parsed and isinstance(parsed, dict):
                normalized = {
                    "name": parsed.get("name"),
                    "email": parsed.get("email"),
                    "phone": parsed.get("phone"),
                    "skills": parsed.get("skills") or [],
                    "experience_years": parsed.get("experience_years"),
                    "education": parsed.get("education") or [],
                    "summary": (parsed.get("summary") or "").strip()
                }
                # If summary present and non-empty -> return it
                if normalized["summary"]:
                    return normalized

                # Otherwise fall through to synthesize below
                groq_parsed = normalized
                break
            else:
                # model returned non-JSON; try again
                continue

        except Exception as e:
            err_str = str(e)
            print(f"GROQ parse attempt {attempt} failed: {err_str}")
            # Fail fast on unrecoverable model errors
            if "model_not_found" in err_str or "model_decommissioned" in err_str:
                print("Unrecoverable model error detected — falling back to rule-based parser.")
                break
            continue

    # If we reach here: either Groq returned no summary or parsing failed.
    # Use rule-based extraction first (or groq_parsed fields if available)
    parsed_source = locals().get("groq_parsed") if "groq_parsed" in locals() else None

    if parsed_source:
        name = parsed_source.get("name")
        skills = parsed_source.get("skills") or []
        years = parsed_source.get("experience_years")
        education = parsed_source.get("education") or []
    else:
        # rule-based fallback
        name = extract_name_candidate(text)
        skills = simple_skill_match(text)
        years = estimate_years_experience(text)
        education = extract_education_lines(text)

    # Build a robust synthesized summary
    parts = []

    # experience part
    if years:
        try:
            years_int = int(years)
            parts.append(f"{years_int} years' experience")
        except Exception:
            # sometimes years is a string or year range — ignore if not int
            parts.append(f"{years} experience" if years else "")

    # skills part: top 3 skills
    if skills:
        top = skills[:3]
        parts.append("skilled in " + ", ".join(top))

    # education part: pick first entry degree/institution
    if education:
        edu0 = education[0]
        if isinstance(edu0, dict):
            deg = edu0.get("degree") or edu0.get("program") or ""
            inst = edu0.get("institution") or edu0.get("school") or ""
            edu_part = " ".join([p for p in [deg, ("from " + inst) if inst else ""] if p]).strip()
            if edu_part:
                parts.append(edu_part)
        elif isinstance(edu0, str):
            parts.append(edu0)

    # If we still have nothing, try to synthesize from raw text's first lines
    if not parts:
        synthesized = build_summary(text, max_chars=240)
        if synthesized:
            summary = synthesized[:240].strip()
        else:
            summary = "No summary available"
    else:
        # join parts into sentences, keep under 240 chars
        summary = ". ".join([p.strip() for p in parts if p]).strip()
        if not summary.endswith("."):
            summary = summary
        if len(summary) > 240:
            summary = summary[:237].rstrip() + "..."

    # Ensure summary is a non-empty string
    if not summary:
        summary = "No summary available"

    # Construct final return object, preferring parsed_source fields where present
    final = {
        "name": name,
        "email": parsed_source.get("email") if parsed_source else extract_contact(text).get("email"),
        "phone": parsed_source.get("phone") if parsed_source else extract_contact(text).get("phone"),
        "skills": skills,
        "experience_years": years,
        "education": education,
        "summary": summary
    }

    return final

# --- FastAPI endpoints -------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "service": "ai-interview-ai-service (groq parser)"}

@app.post("/evaluate")
def evaluate(req: EvalRequest):
    words = len(req.candidate_answer.split())
    llm_score = min(10, max(0, words / 8))
    embedding_score = round(min(1.0, words / 50), 3)
    heuristics_score = 10 if words >= 20 else (5 if words >= 10 else 2)
    final = round(0.7 * llm_score + 0.25 * (embedding_score * 10) + 0.05 * heuristics_score, 2)
    return {
        "score": final,
        "breakdown": {
            "llm_score": round(llm_score, 2),
            "embedding_similarity": embedding_score,
            "heuristics_score": heuristics_score,
        },
        "subscores": {
            "correctness": round(min(10, llm_score), 2),
            "completeness": round(min(10, llm_score * 0.9), 2),
            "clarity": round(min(10, llm_score * 0.9), 2),
        },
        "summary": "Placeholder evaluation. Replace with LLM & embeddings for production.",
        "improvements": ["Be more specific.", "Include an example.", "Structure your answer."],
    }

@app.post("/parse_resume")
async def parse_resume(
    file: UploadFile = File(None),
    s3_url: Optional[str] = Form(None),
    text: Optional[str] = Form(None),
    resume_id: Optional[str] = Form(None),
):
    raw_text = ""

    # 1) file uploaded
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
            except Exception:
                raw_text = ""

    # 2) text provided directly
    elif text:
        raw_text = text

    # 3) s3_url or local path provided
    elif s3_url:
        if s3_url.startswith("http://") or s3_url.startswith("https://"):
            try:
                import requests
                r = requests.get(s3_url, timeout=15)
                if r.status_code == 200:
                    content_type = r.headers.get("content-type", "")
                    if "pdf" in content_type or s3_url.lower().endswith(".pdf"):
                        raw_text = extract_text_from_pdf_bytes(r.content)
                    elif s3_url.lower().endswith(".docx"):
                        raw_text = extract_text_from_docx_bytes(r.content)
                    else:
                        raw_text = r.text
            except Exception:
                raw_text = ""
        else:
            raw_text = extract_text_from_path(s3_url)

    raw_text = (raw_text or "").strip()

    if not raw_text:
        return {"parsed": {"error": "no_text_extracted", "skills": [], "summary": ""}}

    # Use Groq LLM to parse into structured JSON (primary)
    parsed = groq_parse_resume(raw_text)

    return {"parsed": parsed}
