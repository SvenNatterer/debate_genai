"""
test_azure_connection.py
========================
Standalone diagnostic script for your Azure AI Services connection.

Run from your project folder:
    python test_azure_connection.py

What this script does
---------------------
1. Loads your .env file
2. Checks that required variables are present
3. Lists available deployments on your Azure endpoint  <-- finds exact model names
4. Sends a minimal chat request to each model you want to test
5. Reports timing, status codes, and response content

No extra packages required beyond python-dotenv.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import time
import urllib.error
import urllib.request
from openai import AzureOpenAI
import instructor

# ── Load .env ────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[OK] .env loaded via python-dotenv\n")
except ImportError:
    print("[WARN] python-dotenv not installed — reading OS environment only\n")

# ── Read config ───────────────────────────────────────────────────────────────
BASE_URL    = os.getenv("CUSTOM_API_BASE_URL", "").rstrip("/")
API_KEY     = os.getenv("CUSTOM_API_KEY", "")
API_VERSION = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")

# Target the failing model
MODEL_TO_TEST = "Phi-4-mini-reasoning"

from pydantic import BaseModel, Field

class SimpleResponse(BaseModel):
    reply: str = Field(description="A brief reply")

def test_raw_client():
    print(f"\n--- Testing RAW AzureOpenAI Client for {MODEL_TO_TEST} ---")
    client = AzureOpenAI(
        api_key=API_KEY,
        api_version=API_VERSION,
        azure_endpoint=BASE_URL,
        timeout=30.0
    )
    start = time.time()
    try:
        response = client.chat.completions.create(
            model=MODEL_TO_TEST,
            messages=[{"role": "user", "content": "Say 'RAW OK'"}],
            max_tokens=10
        )
        elapsed = time.time() - start
        print(f"[OK] Raw Response: '{response.choices[0].message.content.strip()}' ({elapsed:.2f}s)")
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"[FAIL] Raw Request failed: {e} ({elapsed:.2f}s)")
        return False

from debate_engine_cloud import chat_completion, PhilosopherResponse

def test_full_integration():
    print(f"\n--- Testing FULL INTEGRATION (with Fallback) for {MODEL_TO_TEST} ---")
    
    start = time.time()
    try:
        response = chat_completion(
            "You are a philosopher.",
            "Explain the meaning of life in one sentence.",
            provider="custom",
            model=MODEL_TO_TEST,
            response_model=PhilosopherResponse
        )
        elapsed = time.time() - start
        if isinstance(response, str):
            print(f"[FAIL] Integration failed: {response} ({elapsed:.2f}s)")
            return False
        
        print(f"[OK] Integration Response: '{response.argument}' ({elapsed:.2f}s)")
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"[FAIL] Integration Request failed: {e} ({elapsed:.2f}s)")
        return False

if __name__ == "__main__":
    if not BASE_URL or not API_KEY:
        print("Missing credentials.")
        exit(1)
    
    # test_raw_client()
    integration_ok = test_full_integration()
    
    if integration_ok:
        print("\nCONCLUSION: The fix (MD_JSON + Manual Fallback) works!")
    else:
        print("\nSTILL FAILING: Check the logs.")
