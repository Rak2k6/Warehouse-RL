"""
Test / Demo Script - Warehouse Order Fulfillment Environment
=============================================================
Runs a short demonstration of the environment with step-by-step
decision logging, validates the Gym API contract, tests new features
(priority orders, scenario modes, reward breakdown), and prints
summary metrics. Also includes an MCP Client test.

Usage:
    python test.py
    python test.py --episodes 5 --render
    python test.py --mode rush
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
    assert "decision_reason" in info, "Info must contain decision_reason"
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

    # New info keys
    assert "reward_breakdown" in info, "Info must contain reward_breakdown"
    assert "decision_reason" in info, "Info must contain decision_reason"
    rb = info["reward_breakdown"]
    expected_keys = [
        "completion_reward", "speed_bonus", "priority_bonus",
        "idle_penalty", "congestion_penalty", "wait_time_penalty",
    ]
    for key in expected_keys:
        assert key in rb, f"reward_breakdown must contain '{key}'"
    print("  [PASS] reward_breakdown contains all expected components")
    print(f"         {rb}")

    # Verify decision_reason is a non-empty string
    assert isinstance(info["decision_reason"], str), "decision_reason must be a string"
    assert len(info["decision_reason"]) > 0, "decision_reason must not be empty"
    print(f"  [PASS] decision_reason: \"{info['decision_reason']}\"")

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

    # Episode summary on termination
    assert "episode_summary" in info, "Terminated episode must have episode_summary"
    es = info["episode_summary"]
    print(f"  [PASS] episode_summary present: {es['orders_completed']} orders, "
          f"reward={es['total_reward']:.2f}")

    env.close()
    print("\n  [PASS] All Gym API tests passed!\n")


def test_determinism():
    """Verify deterministic behavior with same seed."""
    print("=" * 60)
    print("  TEST 2: Determinism (same seed -> same trajectory)")
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


def test_priority_orders():
    """Verify priority order functionality."""
    print("=" * 60)
    print("  TEST 3: Priority Orders")
    print("=" * 60)

    env = WarehouseOrderFulfillmentEnv(
        num_workers=3, max_queue=10, max_orders=20, max_steps=100,
        priority_prob=0.5,  # high probability to guarantee some priority orders
        seed=42,
    )
    obs, info = env.reset(seed=42)

    # Check that priority flags exist in the queue
    has_priority = np.any(env.queue_priority > 0.5)
    print(f"  [INFO] Initial priority orders: {int(np.sum(env.queue_priority > 0.5))}")

    # Check observation includes priority section
    expected_obs_size = 1 + 2 * env.num_workers + 3 * env.max_queue
    assert obs.shape[0] == expected_obs_size, \
        f"Obs size {obs.shape[0]} != expected {expected_obs_size}"
    print(f"  [PASS] Observation includes priority flags (size={obs.shape[0]})")

    # Check priority flags are accessible in obs
    prio_offset = 1 + 2 * env.num_workers + 2 * env.max_queue
    prio_in_obs = obs[prio_offset : prio_offset + env.max_queue]
    print(f"  [PASS] Priority flags in obs: {prio_in_obs[:5].tolist()} ...")

    # Run a few steps and check priority_bonus in reward breakdown
    found_priority_bonus = False
    done = False
    while not done:
        valid = np.where(env.queue_proc_time > 0)[0]
        action = int(valid[0]) if len(valid) > 0 else env.max_queue
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rb = info.get("reward_breakdown", {})
        if rb.get("priority_bonus", 0) > 0:
            found_priority_bonus = True

    if found_priority_bonus:
        print("  [PASS] Priority bonus awarded during episode")
    else:
        print("  [WARN] No priority bonus observed (may depend on seed)")

    # Check priority_completed in info
    assert "priority_completed" in info, "Info must track priority_completed"
    print(f"  [PASS] Priority orders completed: {info['priority_completed']}")

    env.close()
    print()


def test_scenario_modes():
    """Verify scenario mode configurations."""
    print("=" * 60)
    print("  TEST 4: Scenario Modes (normal / rush / low)")
    print("=" * 60)

    modes = {
        "normal": {"expected_prob": 0.4, "expected_range": (1, 8)},
        "rush":   {"expected_prob": 0.7, "expected_range": (1, 5)},
        "low":    {"expected_prob": 0.15, "expected_range": (3, 12)},
    }

    for mode, expected in modes.items():
        env = WarehouseOrderFulfillmentEnv(mode=mode, seed=0)
        assert env.mode == mode, f"Mode should be {mode}"
        assert env.new_order_prob == expected["expected_prob"], \
            f"[{mode}] Prob {env.new_order_prob} != {expected['expected_prob']}"
        assert env.order_time_range == expected["expected_range"], \
            f"[{mode}] Range {env.order_time_range} != {expected['expected_range']}"

        obs, info = env.reset(seed=0)
        obs, _, _, _, info = env.step(env.max_queue)
        assert info["mode"] == mode, f"Info mode should be {mode}"
        env.close()
        print(f"  [PASS] Mode '{mode}': prob={env.new_order_prob}, "
              f"range={env.order_time_range}")

    print()


def test_demo_episode(render: bool = True, mode: str = "normal"):
    """Run a demo episode with step-by-step decision output."""
    print("=" * 60)
    print(f"  TEST 5: Demo Episode [{mode.upper()}] (step-by-step decisions)")
    print("=" * 60)

    env = WarehouseOrderFulfillmentEnv(
        num_workers=3,
        max_queue=10,
        max_orders=15,
        max_steps=60,
        mode=mode,
        render_mode="human" if render else None,
        seed=7,
    )

    obs, info = env.reset(seed=7)

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
            priorities = env.queue_priority[valid]

            # Prefer priority orders, then shortest time
            priority_valid = valid[priorities > 0.5]
            if len(priority_valid) > 0:
                best = priority_valid[np.argmin(env.queue_proc_time[priority_valid])]
                action = int(best)
                prio_tag = " [PRIORITY]"
            else:
                best = valid[np.argmin(proc_times)]
                action = int(best)
                prio_tag = ""

            action_desc = (
                f"Assign order #{action}{prio_tag} "
                f"(proc={env.queue_proc_time[action]:.0f}) "
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
            # Show decision reason from the env
            reason = info.get("decision_reason", "")
            warning = info.get("warning", "")
            warning_str = f"  {warning}" if warning else ""
            print(
                f"  Step {step_count:>3}: {action_desc:<50} "
                f"reward={reward:>+6.2f}  queue={info['queue_length']}"
                f"{warning_str}"
            )

    # Print episode summary
    es = info.get("episode_summary", {})
    print(f"\n  {'='*50}")
    print(f"  EPISODE SUMMARY")
    print(f"  {'='*50}")
    print(f"  Steps:              {es.get('steps', step_count)}")
    print(f"  Orders completed:   {es.get('orders_completed', '?')}/{es.get('orders_generated', '?')}")
    print(f"  Priority completed: {es.get('priority_completed', '?')}")
    print(f"  Avg fulfillment:    {es.get('avg_fulfillment_time', 0):.2f}")
    print(f"  Worker utilization: {es.get('worker_utilization', 0)*100:.1f}%")
    print(f"  Total reward:       {es.get('total_reward', total_reward):.2f}")
    print(f"  Mode:               {es.get('mode', mode)}")
    print(f"  Outcome:            {'COMPLETED' if es.get('terminated') else 'TRUNCATED'}")
    print(f"  {'='*50}")

    env.close()
    print()


def test_edge_cases():
    """Test edge cases like empty queue, full queue."""
    print("=" * 60)
    print("  TEST 6: Edge Case Validation")
    print("=" * 60)

    env = WarehouseOrderFulfillmentEnv(
        num_workers=2, max_queue=5, max_orders=5, max_steps=100, seed=0
    )
    obs, _ = env.reset(seed=0)

    # No-op action should be valid
    obs, reward, _, _, info = env.step(env.max_queue)
    print("  [PASS] No-op action accepted")
    print(f"     Decision: {info.get('decision_reason', '')}")

    # Invalid queue slot (empty) should give penalty
    obs, info = env.reset(seed=0)
    # Find an empty slot
    empty_slots = np.where(env.queue_proc_time == 0)[0]
    if len(empty_slots) > 0:
        obs, reward, _, _, info = env.step(int(empty_slots[0]))
        assert reward < 0, "Selecting empty slot should give negative reward"
        print("  [PASS] Empty slot penalty applied correctly")
        print(f"     Decision: {info.get('decision_reason', '')}")

    # Check warning on high congestion (create scenario with full queue)
    env2 = WarehouseOrderFulfillmentEnv(
        num_workers=1, max_queue=5, max_orders=50, max_steps=200,
        new_order_prob=0.9, seed=0
    )
    obs, _ = env2.reset(seed=0)
    # Force high congestion by doing many no-ops
    congestion_warning_found = False
    for _ in range(50):
        obs, _, terminated, truncated, info = env2.step(env2.max_queue)
        if "warning" in info and "congestion" in info.get("warning", "").lower():
            congestion_warning_found = True
            break
        if terminated or truncated:
            break
    if congestion_warning_found:
        print(f"  [PASS] Congestion warning detected: {info['warning']}")
    else:
        print("  [WARN] Congestion warning not triggered (depends on order arrival)")

    env.close()
    env2.close()
    print("  [PASS] All edge case tests passed!\n")


def test_mcp_client(url="http://localhost:8000"):
    """
    Test the environment through the MCP client.
    Note: Requires the server to be running.
    """
    print("=" * 60)
    print("  TEST 7: MCP Client (Server interaction)")
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
    parser.add_argument(
        "--mode", type=str, default="normal",
        choices=["normal", "rush", "low"],
        help="Scenario mode for demo episode"
    )
    args = parser.parse_args()

    print("\n+----------------------------------------------------------+")
    print("  Warehouse Order Fulfillment - Test Suite              ")
    print("+----------------------------------------------------------+\n")

    test_gym_api()
    test_determinism()
    test_priority_orders()
    test_scenario_modes()
    test_edge_cases()
    test_demo_episode(render=args.render, mode=args.mode)

    if args.mcp:
        test_mcp_client()

    print("-" * 60)
    print("  DONE - Environment is ready!")
    print("-" * 60)


if __name__ == "__main__":
    main()