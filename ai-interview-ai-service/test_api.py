import requests, json

API = "http://127.0.0.1:8000/generate_question"
payload = {
    "request_id": "REQ123",
    "session_id": "S1",
    "user_id": "U1",
    "resume_summary": "Built a Python data processing service. Used lists and numeric transforms.",
    "retrieved_chunks": [],
    "conversation": [],
    "question_history": []
}

r = requests.post(API, json=payload, timeout=15)
print("HTTP", r.status_code)
j = r.json()
# pretty print parsed part
print(json.dumps(j.get("parsed"), indent=2))
# If parsed exists, extract test cases:
parsed = j.get("parsed") or {}
cc = parsed.get("coding_challenge") or {}
tcs = cc.get("test_cases")
print("test_cases:", tcs)

