import requests
import json

url = "http://127.0.0.1:8000/run_code"  # adjust port if your FastAPI runs elsewhere

payload = {
    "language": "python",
    "code": """
def solve(x):
    return x + 1
""",
    "stdin": "1",
    "expected_output": "2",
    "test_cases": [
        {"input": "1", "expected": "2"},
        {"input": "5", "expected": "6"}
    ]
}

resp = requests.post(url, json=payload)
print("STATUS:", resp.status_code)

try:
    print(json.dumps(resp.json(), indent=2))
except Exception:
    print("RAW RESPONSE:")
    print(resp.text)
