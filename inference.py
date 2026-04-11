import requests

BASE_URL = "http://localhost:7860"

def main():
    try:
        print("[START] task=warehouse env=openenv model=baseline")

        res = requests.post(f"{BASE_URL}/reset", json={})
        data = res.json()

        obs = data.get("metadata", {}).get("observation", [])
        done = False
        step = 0
        rewards = []

        while not done and step < 50:
            action = 0

            res = requests.post(
                f"{BASE_URL}/step",
                json={"action": action}
            )
            data = res.json()

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
