


import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def test_integration():
    print("🔄 STARTING FULL FLOW INTEGRATION TEST")
    print("========================================")

    # ---------------------------------------------------------
    # STEP 1: GENERATE THE QUESTION
    # ---------------------------------------------------------
    print("\n1️⃣  STEP 1: Generatng Coding Question...")
    
    # Mock history to force Q3 (Coding Challenge)
    history = [
        {"question": "Project Q", "type": "project_discussion", "score": 0.8},
        {"question": "Exp Q", "type": "experience", "score": 0.8}
    ]
    
    gen_payload = {
        "request_id": "test-flow-001",
        "session_id": "session-flow-1",
        "user_id": "user-1",
        "resume_summary": "Experienced Python Developer with DSA skills.",
        "question_history": history,
        "token_budget": 4000
    }

    try:
        resp = requests.post(f"{BASE_URL}/generate_question", json=gen_payload)
        resp.raise_for_status()
        data = resp.json()
        parsed = data.get("parsed", {})
        
        # Validation
        if parsed.get("type") != "coding_challenge":
            print(f"❌ FAILED: Expected 'coding_challenge', got '{parsed.get('type')}'")
            return

        cc = parsed.get("coding_challenge", {})
        starter_code = cc.get("starter_code")
        test_cases = cc.get("test_cases", [])

        if not starter_code or not test_cases:
            print("❌ FAILED: Missing starter_code or test_cases in response.")
            print(json.dumps(parsed, indent=2))
            return

        print(f"✅ Question Generated: {parsed.get('question')[:60]}...")
        print(f"   Starter Code: {starter_code}")
        print(f"   Test Cases Extracted: {len(test_cases)}")

    except Exception as e:
        print(f"❌ CRITICAL ERROR in Step 1: {e}")
        return

    # ---------------------------------------------------------
    # STEP 2: RUN THE CODE (Using Starter Code)
    # ---------------------------------------------------------
    print("\n2️⃣  STEP 2: Testing Execution Endpoint...")
    print("   (Sending starter_code to /run_code. Expecting FAILURE on tests, but SUCCESS on execution)")

    run_payload = {
        "language": "python",
        "code": starter_code,  # We run the empty starter code
        "test_cases": test_cases
    }

    try:
        # Measure latency
        start_time = time.time()
        run_resp = requests.post(f"{BASE_URL}/run_code", json=run_payload)
        latency = time.time() - start_time
        
        run_resp.raise_for_status()
        result = run_resp.json()

        # Validation
        if not result.get("success"):
            # This means the sandbox crashed, not just failed tests
            print("❌ FAILED: Sandbox execution error.")
            print(result)
            return

        print(f"✅ Execution Completed in {latency:.2f}s")
        print(f"   Sandbox Success: {result.get('success')} (API worked)")
        print(f"   All Tests Passed: {result.get('all_passed')} (Expected False for starter code)")
        
        # Print first test result to prove it actually ran
        if result.get("results"):
            first_res = result["results"][0]
            print(f"   [Proof] Test 1 Input: {first_res['input']}")
            print(f"   [Proof] Test 1 Output: {first_res['stdout'].strip() if first_res['stdout'] else 'None'}")
        
        print("\n🎉 INTEGRATION TEST PASSED!")
        print("The backend successfully generated a question, passed it to the frontend (simulated),")
        print("and the execution engine successfully attempted to run it.")

    except Exception as e:
        print(f"❌ CRITICAL ERROR in Step 2: {e}")

if __name__ == "__main__":
    test_integration()