# CareerPilot AI — Adaptive AI Interview Platform

An end-to-end, production-style **mock interview platform** that runs a candidate through Screening → Technical → Behavioral rounds, using multiple LLMs to generate questions, grade answers, run candidate code, and proctor the session via webcam — all in real time.

The system is built as **three independently deployable microservices**:

```
┌─────────────────────┐      ┌──────────────────────┐      ┌───────────────────────────┐
│  ai-interview-       │      │  ai-interview-        │      │  ai-interview-            │
│  frontend            │◄────►│  backend              │◄────►│  ai-service               │
│  (Next.js 16 / React │ REST │  (Node.js / Express 5)│ REST │  (FastAPI / Python)       │
│  19, TypeScript)     │      │  Orchestration + DB   │      │  LLM orchestration, CV,   │
│                      │      │                       │      │  scoring, code sandbox    │
└─────────────────────┘      └──────────┬────────────┘      └───────────────────────────┘
                                          │
                              ┌───────────┴───────────┐
                              │  MongoDB   │  Redis    │
                              │  (state)   │  (locks)  │
                              └────────────┴───────────┘
```

## Why three services, not one?

- **Frontend** only knows how to render UI and capture browser media (camera/mic) — it never talks to an LLM or a model directly.
- **Backend (Node)** is the *system of record*. It owns MongoDB, JWT auth, Redis-based distributed locking, request orchestration, and business rules (round progression, elimination thresholds, latency telemetry). It has **no ML code** — it just calls the AI service over HTTP and persists results.
- **AI service (Python/FastAPI)** is the only place that touches LLMs, computer vision models, and the code sandbox. Isolating it lets it scale independently (it's CPU/GPU-heavy) and keeps heavyweight ML dependencies (OpenCV, DeepFace, YOLOv8) out of the Node process.
---

<img width="946" height="423" alt="image" src="https://github.com/user-attachments/assets/ba71cc9a-e7c5-4f6a-9688-b5d0ac7ec44c" />

## Tech Stack

### 1. Frontend — `ai-interview-frontend`
| Concern | Choice |
|---|---|
| Framework | Next.js 16 (App Router), React 19, TypeScript |
| Styling | Tailwind CSS v4, shadcn/ui + Radix primitives |
| Speech-to-text | Browser **Web Speech API** (`useWebSpeech`) |
| Code editor | Monaco Editor (`@monaco-editor/react`) |
| Whiteboard | Excalidraw (system-design questions) |
| Charts / reports | Recharts, jsPDF + jspdf-autotable (exportable interview report) |
| State | React hooks/context (`AuthContext`, `useInterview`, `useProfile`) — no external state library |

### 2. Backend — `ai-interview-backend`
| Concern | Choice |
|---|---|
| Runtime | Node.js, Express 5 |
| Database | MongoDB via Mongoose (Users, Sessions, QA turns, Resumes, Decisions) |
| Cache / locks | Redis (`redis` client) — distributed locks per session+operation |
| Auth | JWT (`jsonwebtoken`) + bcrypt password hashing, token-version revocation |
| Security | `helmet`, `express-rate-limit`, `express-mongo-sanitize`, Multer with strict MIME/extension allow-list |
| Docs | Swagger UI served from `swagger.yaml` at `/api-docs` |

### 3. AI Service — `ai-interview-ai-service`
| Concern | Choice |
|---|---|
| Framework | FastAPI (Python), Uvicorn, deployed to a Hugging Face Docker Space (port 7860) |
| LLM orchestration | OpenRouter (`Llama-3.3-70B`, `DeepSeek-R1-Distill-70B`, `Qwen-2.5`, `Gemma-3-12B`) with **cascading fallback**, Groq as an emergency net |
| Resume parsing | Groq/OpenRouter LLM-first extraction with a regex-based fallback; `pdfplumber` / `pdfminer.six` / `docx2txt` for text extraction |
| Computer vision | OpenCV (image quality checks), **DeepFace (VGG-Face + MTCNN)** for identity verification, **YOLOv8** (`ultralytics`) for prohibited-object detection |
| Code execution | Glot.io public sandbox API (Python/C++) with retry + backoff |
| Validation | Pydantic v2 schemas for every request/response contract |

---

## High-Level Feature Set

- **Adaptive question generation** — an 8-stage flow (`project_discussion → experience → coding_challenge → project_discussion → coding_challenge → achievement → coding_challenge → conceptual`) driven by resume content, prior answers, and a configurable "company style" (FAANG / Startup / Enterprise).
- **Multi-round elimination pipeline** — Screening → Technical → Behavioral, each with its own pass threshold, min/max question counts, and hard termination rules (instant-fail threshold, consecutive-fail streaks, excellence streaks, max question cap).
- **Gray-zone probing** — when an answer scores in an ambiguous band, the system asks a targeted follow-up ("probe") instead of moving on, to distinguish real understanding from a lucky guess.
- **Live webcam proctoring** — face registration at interview start, then periodic re-verification: face-count check, cosine-distance identity match, and YOLOv8 object detection (e.g. phone) — all returned as `200 OK` with a `verified:false` payload so the frontend can react without treating it as a hard error.
- **In-browser code execution** — candidate code runs against visible + one hidden stress test case via a sandboxed remote execution API, orchestrated by the Node backend.
- **Session-safe concurrency** — every mutating operation (`answer`, `violation`, `proctor`, `start`) is wrapped in a Redis `NX`/`EX` distributed lock with operation-specific timeouts, so two concurrent requests for the same session can't race.
- **Full audit trail** — every question/answer, score, rationale, technical diagnosis (win/gap/fix), and integrity event is persisted in MongoDB, plus round-by-round and final AI-generated feedback with a downloadable PDF report.

---

## Repository Structure

```
AI-interview-assistant-main/
├── ai-interview-frontend/        # Next.js app
│   └── src/app/
│       ├── interview/            # Live interview page, hooks, panels
│       ├── profile/               # Post-interview analytics dashboard
│       ├── resume/                # Resume upload UI
│       ├── Auth/                  # Login / Signup
│       └── components/            # Shared UI (shadcn/ui + custom)
├── ai-interview-backend/
│   ├── src/server.js               # Express app + all REST routes
│   ├── Controller/                 # auth.js, profile.js
│   ├── models/                     # Mongoose schemas (User, Session, QA, Resume, Decision)
│   ├── db/                         # connection.js, redisCache.js, sessionLock.js
│   └── swagger.yaml
└── ai-interview-ai-service/
    ├── main.py                     # FastAPI entrypoint
    ├── core/config.py              # Model lists, round rules, scoring weights, LLM clients
    ├── api/                        # interview_routes, code_routes, proctoring_routes, resume_routes
    ├── services/
    │   ├── common.py                # Shared utilities (JSON extraction, PII redaction, etc.)
    │   ├── decision.py              # Hiring decision + probe-question prompts
    │   ├── prompt_question.py       # Next-question generation prompts
    │   ├── prompt_scoring.py        # Answer scoring / rubric prompts
    │   ├── prompt_feedback.py       # Round + final feedback prompts
    │   ├── projects.py              # Resume project extraction & classification
    │   ├── resume_parser.py         # LLM + regex resume parsing
    │   ├── runtime.py               # Remote code execution sandbox
    │   └── state.py                 # In-memory interview state
    └── models/schemas.py            # Pydantic request/response models
```

---

## Running Locally

```bash
# 1. AI service (Python 3.11+)
cd ai-interview-ai-service
pip install -r requirements.txt
# also: pip install ultralytics deepface scipy google-genai (not pinned in requirements.txt)
uvicorn main:app --reload --port 7860   # requires OPENROUTER_API_KEY in .env

# 2. Backend (Node 18+)
cd ai-interview-backend
npm install
npm run dev                              # requires MONGO_URI, JWT_SECRET, REDIS_URL, AI_URL

# 3. Frontend
cd ai-interview-frontend
npm install
npm run dev                              # http://localhost:3000
```

## API Surface (Backend)

| Route | Purpose |
|---|---|
| `POST /auth/signup` / `/auth/login` | JWT-based auth |
| `POST /process-resume` | Upload resume → forwarded to AI service `/parse_resume` |
| `POST /interview/register-face` | Register the reference face embedding for a session |
| `POST /interview/start` | Create session, verify face, get first question |
| `POST /interview/answer` | Score answer, decide probe vs next question vs round-end vs elimination |
| `POST /interview/proctor` | Per-frame identity/object/multi-face check |
| `POST /interview/violation` | Log an integrity violation (tab switch, etc.) |
| `POST /interview/hint` | AI-generated hint for the current question |
| `POST /interview/end` | Finalize session, generate hire/reject decision |
| `GET /interview/feedback/final/:sessionId` | Full report for the profile dashboard |
| `POST /run-code` | Execute candidate code against test cases |
