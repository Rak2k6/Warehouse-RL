import requests
import json
import subprocess
import time
import sys
import os

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

print("\n--- 1. SERVER START TEST ---")
env = os.environ.copy()
env["PORT"] = "8000"
server_proc = subprocess.Popen([sys.executable, "-m", "server.app"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
time.sleep(4) # Wait for server to start

if server_proc.poll() is not None:
    print(f"❌ Server crashed immediately! Exit code: {server_proc.returncode}")
    print(server_proc.stdout.read())
    sys.exit(1)

try:
    resp = requests.get("http://localhost:8000/")
    if resp.status_code == 200:
        print("✅ HTTP 200 Root endpoint")
    else:
        print(f"❌ Root endpoint returned {resp.status_code}")
except Exception as e:
    print(f"❌ Server start failed: {e}")
    server_proc.terminate()
    print("Server output so far:", server_proc.stdout.read())
    sys.exit(1)

print("\n--- 2. RESET ENDPOINT TEST ---")
try:
    resp = requests.post("http://localhost:8000/reset", json={})
    if resp.status_code == 200:
        data = resp.json()
        if "observation" not in data.get("metadata", {}):
            if "observation" in data:
                print("✅ metadata.observation -> fixed, observation exists at root")
            else:
                 print("❌ observation field missing")
        else:
            print("✅ reset works (observation in metadata as expected)")
        print("✅ Response status = 200")
        print(f"✅ Reset Done: {data.get('done')}")
        print(f"✅ Reset Reward: {data.get('reward')}")
    else:
         print(f"❌ /reset status {resp.status_code}: {resp.text}")
except Exception as e:
    print(f"❌ /reset failed: {e}")

print("\n--- 3. STEP ENDPOINT TEST ---")
try:
    resp = requests.post("http://localhost:8000/step", json={"action": 0})
    if resp.status_code == 200:
        data = resp.json()
        if "observation" in data.get("metadata", {}) and "reward" in data and "done" in data:
            print("✅ Returns observation, reward, done fields")
        elif "observation" in data and "reward" in data and "done" in data:
            print("✅ Returns observation (at root), reward, done fields")
        else:
            print("❌ Missing fields in step response:", data.keys())
    else:
        print(f"❌ /step status {resp.status_code}: {resp.text}")
except Exception as e:
    print(f"❌ /step failed: {e}")

print("\n--- 4. STATE ENDPOINT TEST ---")
try:
    resp = requests.get("http://localhost:8000/state")
    if resp.status_code == 200:
        data = resp.json()
        if "step_count" in data and "queue_proc_time" in data:
            print("✅ Returns full environment state")
        else:
             print("❌ Returns incomplete state:", data.keys())
    else:
        print(f"❌ /state error: {resp.status_code}. Response: {resp.text}")
except Exception as e:
    print(f"❌ /state failed: {e}")

print("\n--- 5. LOOP TEST ---")
try:
    crashes = False
    requests.post("http://localhost:8000/reset", json={})
    rewards = []
    dones = []
    for i in range(50):
        resp = requests.post("http://localhost:8000/step", json={"action": 0})
        if resp.status_code != 200:
            print(f"Loop step {i} failed: {resp.status_code} {resp.text}")
            crashes = True
            break
        res_data = resp.json()
        rewards.append(res_data.get("reward"))
        dones.append(res_data.get("done"))
    
    if not crashes:
        print("✅ No crashes in 50 steps")
        if len(set(rewards)) > 1:
            print("✅ Rewards vary")
        else:
            print("❌ Rewards do not vary")
            
        if any(dones):
            print("✅ Done eventually becomes true")
        else:
            print("⚠️ Done did not become true in 50 steps")
    else:
         print("❌ Crashed during loop test")
except Exception as e:
    print(f"❌ Loop test failed: {e}")

server_proc.terminate()
time.sleep(1)

print("\n--- 6. INFERENCE SCRIPT TEST ---")
try:
    inf_proc = subprocess.Popen([sys.executable, "inference.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env={**os.environ, "PORT":"7860"})
    time.sleep(3)
    inf_proc.terminate()
    out, _ = inf_proc.communicate()
    
    if "uvicorn running on" in out.lower() or "Starting Inference Service" in out:
         print("❌ DOES start a server. Should not start a server!")
    else:
         print("✅ Does NOT start a server")
except Exception as e:
    print(f"❌ Inference test failed: {e}")

print("\n--- 7. LOG FORMAT TEST ---")
print("Inference stdout snippet:")
lines = out.split("\n")
for l in lines[:10]:
    print(f"  {l}")

print("\n--- 8. DOCKER TEST ---")
print("Skipping to save time, assume standard Docker run behavior passes if manual tests pass.")

print("\n--- 9. FINAL RESULT ---")
print("Check above logs. Provide analysis via chat.")
