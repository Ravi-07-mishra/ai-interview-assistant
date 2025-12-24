import requests
import json
import time

# 1. Configuration
API_URL = "http://127.0.0.1:8000/generate_roadmap" # Ensure port matches your running backend (8000 for Python, 4000 for Node)
# Note: If testing the Python service directly, use 8000. If testing via Node, use 4000 and ensure auth is handled or disabled for test.
# Let's assume we hit Python directly for logic testing.

SESSION_ID = f"test-hybrid-{int(time.time())}"

# 2. Made-Up History: "Strong Engineer with a Graph Weakness"
history_data = [
    # STRONG: Project Discussion
    {
        "question": "Tell me about the real-time trading engine you built at GoQuant. How did you handle race conditions?",
        "score": 0.88,
        "feedback": "Excellent explanation of optimistic locking and Redis atomic counters.",
        "type": "project_discussion",
        "result": {"improvement": ""}
    },
    # STRONG: Python Internals
    {
        "question": "How does Python's Global Interpreter Lock (GIL) impact multi-threaded performance?",
        "score": 0.92,
        "feedback": "Perfect understanding of CPU-bound vs I/O-bound threading constraints.",
        "type": "conceptual",
        "result": {"improvement": ""}
    },
    # STRONG: System Design
    {
        "question": "Design a URL Shortener like Bit.ly. How would you scale reads vs writes?",
        "score": 0.85,
        "feedback": "Good use of caching strategies and database sharding keys.",
        "type": "system_design",
        "result": {"improvement": ""}
    },
    # ❌ WEAK: The "Gap" (Graph Algorithms)
    {
        "question": "Given a directed graph, write a function to detect if a cycle exists.",
        "score": 0.40,
        "feedback": "You failed to correctly implement the DFS recursion stack. You missed the 'visited' vs 'recursion_stack' distinction.",
        "type": "coding_challenge",
        "result": {"improvement": "Review Graph Theory, specifically Cycle Detection using DFS colors (White/Gray/Black sets)."}
    },
    # STRONG: Behavioral
    {
        "question": "Tell me about a time you disagreed with a PM.",
        "score": 0.80,
        "feedback": "Good use of STAR method.",
        "type": "behavioral",
        "result": {"improvement": ""}
    }
]

def test_roadmap():
    print(f"🚀 Sending 'Hybrid Candidate' profile to {API_URL}...")
    print(f"📊 Profile: Strong Python/System Design, Weak Graphs.")
    
    payload = {
        "session_id": SESSION_ID,
        "user_id": "test_user_hybrid",
        "question_history": history_data
    }

    try:
        response = requests.post(API_URL, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print("\n✅ SUCCESS! Advanced Roadmap Generated:")
            print("="*60)
            
            # Extract Roadmap Details (Handling potential nesting)
            roadmap = data.get("roadmap", {})
            if "weekly_plan" not in roadmap and "roadmap" in roadmap:
                 roadmap = roadmap["roadmap"]

            print(f"📌 ASSESSMENT: {roadmap.get('overall_assessment')}")
            
            print("\n📅 WEEKLY PLAN:")
            weekly_plan = roadmap.get("weekly_plan", [])
            
            for week in weekly_plan:
                print(f"\n🔹 Week {week.get('week')}: {week.get('theme')}")
                print(f"   Goals: {', '.join(week.get('goals', []))}")
                for task in week.get("daily_tasks", []):
                    print(f"   - {task.get('activity')}")
                    for res in task.get("resources", []):
                        print(f"     🔗 {res.get('title')} -> {res.get('url')}")
            
            print("="*60)
            print("📈 METRICS ANALYSIS (Internal):")
            print(json.dumps(data.get("metrics"), indent=2))
            
        else:
            print(f"❌ Failed: {response.status_code}")
            print(response.text)

    except Exception as e:
        print(f"❌ Connection Error: {e}")
        print("Make sure your Python 'main.py' is running on port 8000.")

if __name__ == "__main__":
    test_roadmap()