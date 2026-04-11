import os
import requests
from openai import OpenAI

BASE_URL = "http://localhost:7860"

def main():
    try:
        # Debug Logs
        api_base = os.environ.get("API_BASE_URL")
        api_key = os.environ.get("API_KEY")
        model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")
        
        print(f"DEBUG: API_BASE_URL: {api_base}")
        print(f"DEBUG: API_KEY exists: {bool(api_key)}")
        print(f"DEBUG: MODEL_NAME: {model_name}")

        print("[START] task=warehouse env=openenv model=baseline")

        # Support Dual Mode: Initialize client only if env vars exist
        client = None
        if api_base and api_key:
            client = OpenAI(
                base_url=api_base,
                api_key=api_key
            )
        else:
            print("WARNING: LLM environment variables missing. Falling back to heuristic defaults.")

        res = requests.post(f"{BASE_URL}/reset", json={})
        data = res.json()

        obs = data.get("metadata", {}).get("observation", [])
        done = False
        step = 0
        rewards = []

        while not done and step < 50:
            action = 0
            
            # Safe LLM Call
            if client:
                try:
                    prompt = f"Observation: {obs}. What is the next action? Reply with a single integer."
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0
                    )
                    action = int(response.choices[0].message.content.strip())
                except Exception as e:
                    print(f"DEBUG: LLM Call failed: {e}")
                    action = 0

            res = requests.post(
                f"{BASE_URL}/step",
                json={"action": action}
            )
            data = res.json()

            # Update observation for the next prompt if available
            if "observation" in data:
                obs = data["observation"]
            elif "metadata" in data and "observation" in data["metadata"]:
                obs = data["metadata"]["observation"]

            reward = data.get("reward", 0.0)
            done = data.get("done", False)

            rewards.append(reward)

            print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null")

            step += 1

        score = sum(rewards)

        print(f"[END] success={str(done).lower()} steps={step} score={score:.2f} rewards={','.join([f'{r:.2f}' for r in rewards])}")

    except Exception as e:
        print(f"[END] success=false steps=0 score=0.00 rewards= error={e}")

if __name__ == "__main__":
    main()
