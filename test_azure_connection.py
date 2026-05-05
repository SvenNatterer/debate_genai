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

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

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

# Models you want to test — must match deployment names exactly on Azure
MODELS_TO_TEST = [   
    "gpt-5-chat",
    "gpt-4.1-mini",
    "DeepSeek-V3.2",
    "mistral-Large-3",
    "mistral-small-2503",
    "Llama-4-Maverick-17B-128E-Instruct-FP8",
    "Llama-3.3-70B-Instruct",
    "Phi-4-mini-reasoning", 
]

SEPARATOR = "─" * 60


def print_section(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def make_request(url: str, payload: dict | None = None, method: str = "GET") -> tuple:
    """
    Makes an HTTP request to Azure.
    Returns (status_code, response_body_dict, elapsed_seconds, error_message).
    """
    headers = {
        "api-key":      API_KEY,
        "Content-Type": "application/json",
    }

    data = json.dumps(payload).encode("utf-8") if payload else None
    req  = urllib.request.Request(url, data=data, headers=headers, method=method)

    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            elapsed = time.time() - start
            body    = json.loads(resp.read().decode("utf-8"))
            return resp.status, body, elapsed, None
    except urllib.error.HTTPError as exc:
        elapsed = time.time() - start
        try:
            body = json.loads(exc.read().decode("utf-8"))
            msg  = body.get("error", {}).get("message", str(body))
        except Exception:
            msg = exc.reason
        return exc.code, {}, elapsed, f"HTTP {exc.code}: {msg}"
    except urllib.error.URLError as exc:
        return 0, {}, time.time() - start, f"Connection error: {exc.reason}"
    except TimeoutError:
        return 0, {}, 20.0, "Timed out after 20 seconds"


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Environment check
# ═══════════════════════════════════════════════════════════════════════════════
print_section("STEP 1: Environment variables")

ok = True
if BASE_URL:
    print(f"  [OK] CUSTOM_API_BASE_URL = {BASE_URL}")
else:
    print("  [FAIL] CUSTOM_API_BASE_URL is not set")
    ok = False

if API_KEY:
    masked = API_KEY[:4] + "*" * (len(API_KEY) - 8) + API_KEY[-4:]
    print(f"  [OK] CUSTOM_API_KEY      = {masked}")
else:
    print("  [FAIL] CUSTOM_API_KEY is not set")
    ok = False

print(f"  [INFO] API version       = {API_VERSION}")

if not ok:
    print("\n  Cannot continue — fix your .env file first.")
    raise SystemExit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — List available deployments
# ═══════════════════════════════════════════════════════════════════════════════
print_section("STEP 2: Available deployments on your Azure endpoint")

deployments_url = f"{BASE_URL}/openai/deployments?api-version={API_VERSION}"
print(f"  GET {deployments_url}\n")

status, body, elapsed, error = make_request(deployments_url)

if error:
    print(f"  [FAIL] {error}  ({elapsed:.1f}s)")
    print("\n  Could not list deployments. Possible reasons:")
    print("  - Wrong CUSTOM_API_BASE_URL")
    print("  - API key has no permission to list deployments")
    print("  - Azure endpoint uses a different path (rare)")
    print("\n  Continuing to model tests anyway...\n")
    available_deployments = []
else:
    deployments = body.get("value", [])
    available_deployments = [d.get("id", d.get("model", "")) for d in deployments]
    print(f"  [OK] Found {len(deployments)} deployment(s) in {elapsed:.2f}s:\n")
    for d in deployments:
        name   = d.get("id") or d.get("model") or "(unnamed)"
        status_d = d.get("status", "")
        model  = d.get("model", "")
        print(f"    • {name:<45} status={status_d}  model={model}")
    if not deployments:
        print("  (none returned — the endpoint may not support this list API)")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Send a minimal chat request to each model
# ═══════════════════════════════════════════════════════════════════════════════
print_section("STEP 3: Chat completion test per model")

SIMPLE_PAYLOAD = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant. Be very brief."},
        {"role": "user",   "content": "Reply with exactly: OK"},
    ],
    "temperature": 0,
    "max_tokens": 10,
}

results = []

for model in MODELS_TO_TEST:
    url = (
        f"{BASE_URL}/openai/deployments/{model}/chat/completions"
        f"?api-version={API_VERSION}"
    )
    print(f"\n  Testing: {model}")
    print(f"  URL:     {url}")

    status, body, elapsed, error = make_request(url, payload=SIMPLE_PAYLOAD, method="POST")

    if error:
        print(f"  [FAIL]  {error}  ({elapsed:.1f}s)")
        results.append((model, False, error, elapsed))
    else:
        choices = body.get("choices", [])
        content = choices[0].get("message", {}).get("content", "") if choices else ""
        print(f"  [OK]    Response: '{content.strip()}'  ({elapsed:.2f}s)")
        results.append((model, True, content.strip(), elapsed))


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Summary
# ═══════════════════════════════════════════════════════════════════════════════
print_section("STEP 4: Summary")

passed = [r for r in results if r[1]]
failed = [r for r in results if not r[1]]

print(f"\n  Passed: {len(passed)} / {len(results)}\n")

for model, ok, msg, elapsed in results:
    icon = "[OK]  " if ok else "[FAIL]"
    print(f"  {icon}  {model:<50} {elapsed:.1f}s")
    if not ok:
        print(f"         ↳ {msg}")

if failed:
    print("\n  Common fixes for failing models:")
    print("  - Model name must match the deployment name exactly (case-sensitive)")
    print("  - Use STEP 2 output above to see exact names")
    print("  - Some models may not be deployed on your Azure resource")
    print("  - Try a different API version: set AZURE_API_VERSION=2024-02-01 in .env")

if passed:
    print("\n  Working model names to use in ui.py MODEL_OPTIONS:")
    for model, ok, msg, elapsed in passed:
        print(f"    (\"custom\", \"{model}\")")

print(f"\n{SEPARATOR}\n")
