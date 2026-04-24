import requests
import json

payload = {
    "intent": "create_blog",
    "topic": "police check procedures and background verification policies in the modern workplace",
    "tone": "clear_professional",
    "audience": "general audience",
    "length": "medium",
    "custom_instructions": ""
}

try:
    resp = requests.post("http://localhost:8000/api/chat", json=payload)
    print(f"Status: {resp.status_code}")
    
    data = resp.json()
    print("KEYS IN RESPONSE:", list(data.keys()))
    print("--- DRAFT ---")
    print(data.get("draft", "NO DRAFT KEY"))
    print("--- OUTLINE ---")
    print(data.get("outline", "NO OUTLINE KEY"))
except Exception as e:
    print("Error:", e)
