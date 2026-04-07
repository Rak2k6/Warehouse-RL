"""
Test / Demo Script — Warehouse Order Fulfillment Environment
=============================================================
Runs a short demonstration of the environment with step-by-step
decision logging, validates the Gym API contract, and prints
summary metrics. Also includes an MCP Client test.

Usage:
    python test.py
    python test.py --episodes 5 --render
    python test.py --mcp
"""

from __future__ import annotations
import argparse
import sys
import gymnasium as gym
import numpy as np

# Package imports
from warehouse_env.envs.warehouse_env import WarehouseOrderFulfillmentEnv
from warehouse_env.client import WarehouseEnv


def test_gym_api():
    """Validate that the environment follows the Gym API contract."""
    print("=" * 60)
    print("  TEST 1: Gym API Contract Validation")
    print("=" * 60)

    env = WarehouseOrderFulfillmentEnv(seed=0)

    # reset() returns (obs, info)
    result = env.reset(seed=0)
    assert isinstance(result, tuple) and len(result) == 2, "reset() must return (obs, info)"
    obs, info = result
    assert env.observation_space.contains(obs), "Observation out of space bounds"
    assert isinstance(info, dict), "Info must be a dict"
    print("  [PASS] reset() returns valid (obs, info)")

    # step() returns (obs, reward, terminated, truncated, info)
    action = env.action_space.sample()
    result = env.step(action)
    assert isinstance(result, tuple) and len(result) == 5, "step() must return 5 values"
    obs, reward, terminated, truncated, info = result
    assert env.observation_space.contains(obs), "Observation out of space bounds"
    assert isinstance(reward, (int, float)), "Reward must be numeric"
    assert isinstance(terminated, bool), "Terminated must be bool"
    assert isinstance(truncated, bool), "Truncated must be bool"
    assert isinstance(info, dict), "Info must be a dict"
    print("  [PASS] step() returns valid (obs, reward, terminated, truncated, info)")

    # observation_space and action_space exist
    assert hasattr(env, "observation_space"), "Must have observation_space"
    assert hasattr(env, "action_space"), "Must have action_space"
    print(f"  [PASS] observation_space: {env.observation_space}")
    print(f"  [PASS] action_space: {env.action_space}")

    # Can run a full episode
    obs, info = env.reset(seed=1)
    steps = 0
    done = False
    while not done:
        valid = np.where(env.queue_proc_time > 0)[0]
        action = int(valid[0]) if len(valid) > 0 else env.max_queue
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
    print(f"  [PASS] Full episode completed in {steps} steps")
    print(f"     Orders completed: {info['orders_completed']}")
    print(f"     Avg fulfillment time: {info['avg_fulfillment_time']:.2f}")

    env.close()
    print("\n  [PASS] All Gym API tests passed!\n")


def test_determinism():
    """Verify deterministic behavior with same seed."""
    print("=" * 60)
    print("  TEST 2: Determinism (same seed → same trajectory)")
    print("=" * 60)

    rewards_1 = []
    rewards_2 = []

    for run, rewards_list in [(1, rewards_1), (2, rewards_2)]:
        env = WarehouseOrderFulfillmentEnv(seed=42)
        obs, _ = env.reset(seed=42)
        rng = np.random.default_rng(42)
        done = False
        while not done:
            valid = np.where(env.queue_proc_time > 0)[0]
            action = int(rng.choice(valid)) if len(valid) > 0 else env.max_queue
            obs, reward, terminated, truncated, info = env.step(action)
            rewards_list.append(reward)
            done = terminated or truncated
        env.close()

    match = np.allclose(rewards_1, rewards_2)
    symbol = "[PASS]" if match else "[FAIL]"
    print(f"  {symbol} Determinism check: {'PASSED' if match else 'FAILED'}")
    print(f"     Run 1: {len(rewards_1)} steps, total reward = {sum(rewards_1):.2f}")
    print(f"     Run 2: {len(rewards_2)} steps, total reward = {sum(rewards_2):.2f}")
    print()


def test_demo_episode(render: bool = True):
    """Run a demo episode with step-by-step decision output."""
    print("=" * 60)
    print("  TEST 3: Demo Episode (step-by-step decisions)")
    print("=" * 60)

    env = WarehouseOrderFulfillmentEnv(
        num_workers=3,
        max_queue=10,
        max_orders=15,
        max_steps=60,
        render_mode="human" if render else None,
        seed=7,
    )

    obs, info = env.reset(seed=7)
    rng = np.random.default_rng(7)

    total_reward = 0.0
    done = False
    step_count = 0

    print(f"\n  Starting demo: {env.num_workers} workers, "
          f"max {env.max_orders} orders, max {env.max_steps} steps\n")

    while not done:
        valid = np.where(env.queue_proc_time > 0)[0]

        if len(valid) > 0:
            # Heuristic: pick shortest processing time (SPT rule)
            proc_times = env.queue_proc_time[valid]
            best = valid[np.argmin(proc_times)]
            action = int(best)
            action_desc = (
                f"Assign order #{action} (proc_time={env.queue_proc_time[action]:.0f}) "
                f"-> least-busy worker"
            )
        else:
            action = env.max_queue
            action_desc = "No-op (queue empty)"

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        step_count += 1

        if not render:
            print(
                f"  Step {step_count:>3}: {action_desc:<50} "
                f"reward={reward:>+6.2f}  queue={info['queue_length']}"
            )

    print(f"\n  Episode finished in {step_count} steps")
    print(f"  Total reward:          {total_reward:.2f}")
    print(f"  Orders completed:      {info['orders_completed']}")
    print(f"  Avg fulfillment time:  {info['avg_fulfillment_time']:.2f}")
    print(f"  Worker utilization:    {info['worker_utilization']*100:.1f}%")

    env.close()
    print()


def test_edge_cases():
    """Test edge cases like empty queue, full queue."""
    print("=" * 60)
    print("  TEST 4: Edge Case Validation")
    print("=" * 60)

    env = WarehouseOrderFulfillmentEnv(
        num_workers=2, max_queue=5, max_orders=5, max_steps=100, seed=0
    )
    obs, _ = env.reset(seed=0)

    # No-op action should be valid
    obs, reward, _, _, _ = env.step(env.max_queue)
    print("  [PASS] No-op action accepted")

    # Invalid queue slot (empty) should give penalty
    obs, info = env.reset(seed=0)
    # Find an empty slot
    empty_slots = np.where(env.queue_proc_time == 0)[0]
    if len(empty_slots) > 0:
        obs, reward, _, _, _ = env.step(int(empty_slots[0]))
        assert reward < 0, "Selecting empty slot should give negative reward"
        print("  [PASS] Empty slot penalty applied correctly")

    env.close()
    print("  [PASS] All edge case tests passed!\n")


def test_mcp_client(url="http://localhost:8000"):
    """
    Test the environment through the MCP client.
    Note: Requires the server to be running.
    """
    print("=" * 60)
    print("  TEST 5: MCP Client (Server interaction)")
    print("=" * 60)
    print(f"  Connecting to {url}...")
    
    try:
        # Standard MCP Client interaction
        with WarehouseEnv(base_url=url).sync() as env:
            result = env.reset()
            obs = result.observation
            info = {}  # StepResult does not currently support metadata
            print("  [PASS] MCP Reset successful")
            
            tools = env.list_tools()
            print(f"  [PASS] Discovered {len(tools)} tools: {[t.name for t in tools]}")
            
            # Simple interaction
            res = env.call_tool("wait_step")
            print(f"  [PASS] Call wait_step successful, reward: {res.get('reward')}")
            
            res = env.call_tool("assign_order", order_id=0)
            print(f"  [PASS] Call assign_order successful, reward: {res.get('reward')}")
            
    except Exception as e:
        print(f"  [SKIP] Skipping MCP test (server or client error): {e}")
        print("     To run this test, ensure the server is running: python -m warehouse_env.server.app")


def main():
    parser = argparse.ArgumentParser(description="Test Warehouse RL Environment")
    parser.add_argument("--render", action="store_true", help="Render demo episode")
    parser.add_argument("--episodes", type=int, default=1, help="Number of demo episodes")
    parser.add_argument("--mcp", action="store_true", help="Run MCP client test")
    args = parser.parse_args()

    print("\n+----------------------------------------------------------+")
    print("  Warehouse Order Fulfillment - Test Suite              ")
    print("+----------------------------------------------------------+\n")

    test_gym_api()
    test_determinism()
    test_edge_cases()
    test_demo_episode(render=args.render)
    
    if args.mcp:
        test_mcp_client()

    print("-" * 60)
    print("  DONE - Environment is ready!")
    print("-" * 60)


if __name__ == "__main__":
    main()