import os
import re
import requests
from openai import OpenAI

# ── Required Environment Variables (spec §3) ────────────────────────────────
# API_BASE_URL and MODEL_NAME must have defaults.
# HF_TOKEN is mandatory — no default, raise if missing.
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4.1-mini")
HF_TOKEN     = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# ── OpenAI Client (spec §2: must use OpenAI Client, HF_TOKEN as api_key) ────
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

# ── Constants ────────────────────────────────────────────────────────────────
ENV_BASE_URL = "http://localhost:7860"
MAX_QUEUE    = 20    # action=MAX_QUEUE is always-valid wait/no-op
MAX_STEPS    = 200   # hard safety cap


def parse_action(content: str) -> int:
    """Extract first valid integer from LLM response. Falls back to no-op."""
    match = re.search(r"\b(\d+)\b", content or "")
    if match:
        action = int(match.group(1))
        if 0 <= action <= MAX_QUEUE:
            return action
    return MAX_QUEUE  # safe no-op: always accepted by environment


def get_llm_action(obs: list) -> tuple[int, str | None]:
    """
    Call the LLM proxy and return (action, error_string|None).
    error_string is used verbatim in the [STEP] error= field.
    """
    prompt = (
        f"You control a warehouse order fulfillment system.\n"
        f"Observation: {obs}\n"
        f"Valid actions: 0-{MAX_QUEUE - 1} (assign that queue slot), "
        f"{MAX_QUEUE} (wait/no-op).\n"
        f"Prefer priority orders first, then shortest processing time.\n"
        f"Reply with ONE integer only."
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=10,
    )
    content = response.choices[0].message.content.strip()
    return parse_action(content), None


def main():
    rewards    = []
    step       = 0
    done       = False
    last_error = None

    # ── [START] ──────────────────────────────────────────────────────────────
    print(f"[START] task=warehouse env=openenv model={MODEL_NAME}", flush=True)

    # ── Reset environment ────────────────────────────────────────────────────
    try:
        res = requests.post(f"{ENV_BASE_URL}/reset", json={}, timeout=10)
        res.raise_for_status()
        data = res.json()
        obs  = data.get("metadata", {}).get("observation", [])
    except Exception as e:
        print(
            f"[END] success=false steps=0 rewards= error={e}",
            flush=True,
        )
        return

    # ── Episode loop ─────────────────────────────────────────────────────────
    while not done and step < MAX_STEPS:
        # LLM action selection
        try:
            action, last_error = get_llm_action(obs)
        except Exception as e:
            action     = MAX_QUEUE
            last_error = str(e)

        # Send action to environment
        try:
            res = requests.post(
                f"{ENV_BASE_URL}/step",
                json={"action": action},
                timeout=10,
            )
            res.raise_for_status()
            data = res.json()
        except Exception as e:
            print(
                f"[END] success=false steps={step} "
                f"rewards={','.join(f'{r:.2f}' for r in rewards)} "
                f"error={e}",
                flush=True,
            )
            return

        obs         = data.get("metadata", {}).get("observation", [])
        reward      = float(data.get("reward", 0.0))
        done        = bool(data.get("done", False))
        error_field = last_error if last_error else "null"

        rewards.append(reward)
        step += 1

        # ── [STEP] ────────────────────────────────────────────────────────
        print(
            f"[STEP] step={step} action={action} reward={reward:.2f} "
            f"done={str(done).lower()} error={error_field}",
            flush=True,
        )

    # ── [END] ────────────────────────────────────────────────────────────────
    print(
        f"[END] success={str(done).lower()} steps={step} "
        f"rewards={','.join(f'{r:.2f}' for r in rewards)}",
        flush=True,
    )


if __name__ == "__main__":
    main()