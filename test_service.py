import requests
import time
import multiprocessing
import traceback
import sys

def run_server():
    from inference import main
    sys.argv = ["inference.py"]
    try:
        main()
    except Exception as e:
        print(f"Server crash: {e}")

if __name__ == "__main__":
    print("--- Starting Inference Service Test ---")
    
    p = multiprocessing.Process(target=run_server)
    p.start()
    
    # Poll until server is up
    server_up = False
    for i in range(20):
        try:
            resp = requests.get("http://localhost:7860/", timeout=1)
            if resp.status_code == 200:
                server_up = True
                break
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
        
    if not server_up:
        print("[ERROR] Server failed to start or did not respond in time.")
        p.terminate()
        p.join()
        sys.exit(1)
        
    print("[SUCCESS] Server is online.")
    
    try:
        # Test 1: Healthcheck on GET /
        print("\n[Test 1] Testing GET / (Healthcheck)")
        resp = requests.get("http://localhost:7860/")
        print(f"  Result: {resp.status_code} - {resp.json()}")
        assert resp.status_code == 200
        
        # Test 2: Invalid Predict Body
        print("\n[Test 2] Testing POST /predict with invalid JSON body")
        resp = requests.post("http://localhost:7860/predict", json={})
        print(f"  Result: {resp.status_code} - {resp.json()}")
        assert resp.status_code == 200
        action = resp.json().get("action")
        assert isinstance(action, int)
        assert 0 <= action <= 20 
        # Test 3: Valid Predict Observation
        print("\n[Test 3] Testing POST /predict with a valid observation array")
        obs = [0.1] + [0]*4 + [0]*4 + [0, 0, 2, 0] + [0]*16 + [0]*20 + [0, 0, 1, 0] + [0]*16
        
        resp = requests.post("http://localhost:7860/predict", json={"observation": obs})
        print(f"  Result: {resp.status_code} - {resp.json()}")
        assert resp.status_code == 200
        action = resp.json().get("action")
        assert isinstance(action, int)
        assert 0 <= action <= 20        
        print("\n[SUCCESS] All inference tests passed.")

        print("\n[Test 4] Timeout-safe request")

        resp = requests.post(
            "http://localhost:7860/predict",
            json={"observation": obs},
            timeout=2
        )

        print(f"  Result: {resp.status_code} - {resp.json()}")
        assert resp.status_code == 200
        
        print("\n[Test 5] Malformed request")

        resp = requests.post(
            "http://localhost:7860/predict",
            data="invalid-json",
            timeout=2
        )

        print(f"  Result: {resp.status_code} - {resp.text}")
        assert resp.status_code == 200
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        traceback.print_exc()
        
    finally:
        print("\n--- Shutting down test server ---")
        p.terminate()
        p.join()
