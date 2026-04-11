import requests
import time
import multiprocessing
import sys
import traceback

BASE_URL = "http://localhost:8000"


def run_server():
    from server.app import main
    try:
        main()
    except Exception as e:
        print(f"[CRASH] Server failed: {e}")


if __name__ == "__main__":
    print("🚀 Starting Environment Server Test...")

    p = multiprocessing.Process(target=run_server)
    p.start()

    # Wait for server
    server_up = False
    for i in range(20):
        try:
            r = requests.get(BASE_URL, timeout=1)
            if r.status_code == 200:
                server_up = True
                break
        except:
            pass
        time.sleep(1)

    if not server_up:
        print("❌ Server did not start")
        p.terminate()
        sys.exit(1)

    print("✅ Server is online")

    try:
        # -----------------------------
        # TEST 1: RESET
        # -----------------------------
        print("\n[Test 1] POST /reset")

        r = requests.post(f"{BASE_URL}/reset", timeout=3)
        print("Response:", r.status_code, r.text)

        assert r.status_code == 200
        data = r.json()

        assert "observation" in data
        assert "info" in data
        assert isinstance(data["info"], dict)

        obs = data["observation"]
        assert obs is not None

        print("✅ RESET passed")

        # -----------------------------
        # TEST 2: STEP
        # -----------------------------
        print("\n[Test 2] POST /step")

        action_payload = {"action": 0}

        r = requests.post(
            f"{BASE_URL}/step",
            json=action_payload,
            timeout=3
        )

        print("Response:", r.status_code, r.text)

        assert r.status_code == 200
        data = r.json()

        # Required fields
        assert "observation" in data
        assert "reward" in data
        assert "done" in data
        assert "info" in data

        # Type checks
        assert isinstance(data["reward"], (int, float))
        assert isinstance(data["done"], bool)
        assert isinstance(data["info"], dict)

        print("✅ STEP passed")

        # -----------------------------
        # TEST 3: MULTI STEP
        # -----------------------------
        print("\n[Test 3] Multi-step consistency")

        for i in range(5):
            r = requests.post(
                f"{BASE_URL}/step",
                json={"action": 0},
                timeout=3
            )
            assert r.status_code == 200
            data = r.json()
            if data["done"]:
                break

        print("✅ Multi-step passed")

        print("\n🎉 ALL ENV TESTS PASSED")

    except Exception as e:
        print("\n❌ TEST FAILED:", e)
        traceback.print_exc()

    finally:
        print("\n🛑 Shutting down server")
        p.terminate()
        p.join()