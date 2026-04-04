import requests
import random

class TrafficClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url
        
    def reset(self):
        """Sends a reset request to the API."""
        response = requests.post(f"{self.base_url}/reset")
        response.raise_for_status()
        return response.json()
        
    def step(self, action: int):
        """Sends an action step to the API."""
        response = requests.post(
            f"{self.base_url}/step", 
            json={"action": action}
        )
        response.raise_for_status()
        return response.json()

def total_cars(state) -> int:
    """Returns the total number of cars across all directions."""
    return sum(state.values())

def compute_reward(state):
    """Calculates the reward based on total cars and traffic imbalance."""
    total = sum(state.values())
    ns_traffic = state.get("north", 0) + state.get("south", 0)
    ew_traffic = state.get("east", 0) + state.get("west", 0)
    imbalance = abs(ns_traffic - ew_traffic)
    
    total_penalty = -total
    imbalance_penalty = -0.5 * imbalance
    final_reward = total_penalty + imbalance_penalty
    
    return total_penalty, imbalance_penalty, final_reward

def smart_policy(state, prev_action: int = None) -> int:
    """Intelligently chooses an action based on traffic, minimizing unnecessary switching."""
    ns_traffic = state.get("north", 0) + state.get("south", 0)
    ew_traffic = state.get("east", 0) + state.get("west", 0)
    
    imbalance = abs(ns_traffic - ew_traffic)
    
    # Keep the previous signal if the difference is negligible
    if prev_action is not None and imbalance < 3:
        return prev_action
    
    if ns_traffic > ew_traffic:
        return 0  # North-South green
    else:
        return 1  # East-West green

def select_action(state, epsilon, prev_action=None) -> tuple[int, str]:
    """Selects an action using epsilon-greedy strategy."""
    if random.random() < epsilon:
        return random.choice([0, 1]), "Exploration"
    else:
        return smart_policy(state, prev_action), "Exploitation"

def run_simulation(client: TrafficClient, policy_type: str = "random", steps: int = 10, start_epsilon: float = 0.3) -> float:
    """Runs a full simulation episode using a specific behavior policy."""
    print(f"\n{'='*15} Running {policy_type.capitalize()} Policy {'='*15}")
    state = client.reset()
    
    print(f"Initial State: {state}")
    total_cars_sum = 0
    prev_action = None
    epsilon = start_epsilon
    
    for i in range(1, steps + 1):
        # Calculate traffic before taking the action for logging
        ns_traffic = state.get("north", 0) + state.get("south", 0)
        ew_traffic = state.get("east", 0) + state.get("west", 0)
        
        # Decide action based on policy
        action_type = "Determined"
        if policy_type == "epsilon-greedy":
            action, action_type = select_action(state, epsilon, prev_action)
        elif policy_type == "smart":
            action = smart_policy(state, prev_action)
        else:
            action = random.choice([0, 1])
            action_type = "Random"
            
        action_name = "North-South Green" if action == 0 else "East-West Green"
        
        # Output Debug logging (BEFORE action for epsilon-greedy context)
        if policy_type == "epsilon-greedy":
            print(f"\nStep: {i} | Epsilon: {epsilon:.3f} | Mode: {action_type}")
        else:
            print(f"\nStep: {i}")

        # Execute action in environment
        state = client.step(action)
        prev_action = action
        
        # Compute performance metrics for the step
        cars = total_cars(state)
        total_cars_sum += cars
        
        total_penalty, imbalance_penalty, final_reward = compute_reward(state)
        
        # Output Debug logging
        print(f"NS Traffic vs EW Traffic: {ns_traffic} vs {ew_traffic}")
        print(f"Chosen Action: {action_name}")
        print(f"Updated State: {state}")
        print(f"Reward Breakdown:")
        print(f"  Total Cars Penalty: {total_penalty}")
        print(f"  Imbalance Penalty: {imbalance_penalty}")
        print(f"  Final Reward: {final_reward}")

        # Epsilon decay
        if policy_type == "epsilon-greedy":
            epsilon = max(0.05, epsilon * 0.95)
        
    average_cars = total_cars_sum / steps
    print(f"\n-> Average total cars using {policy_type} policy: {average_cars:.2f}")
    return average_cars

if __name__ == "__main__":
    client = TrafficClient()
    
    print("Starting simulation evaluation...")
    
    # Run the simulation with random actions
    avg_random = run_simulation(client, policy_type="random", steps=15)
    
    # Run the simulation with epsilon-greedy logic
    avg_egreedy = run_simulation(client, policy_type="epsilon-greedy", steps=15, start_epsilon=0.3)
    
    # Print the final comparison
    print("\n" + "="*40)
    print("         PERFORMANCE COMPARISON")
    print("="*40)
    print(f"Random Actions:    {avg_random:.2f} avg cars waiting")
    print(f"E-Greedy Policy:   {avg_egreedy:.2f} avg cars waiting")
    
    if avg_egreedy < avg_random:
        print("Result: E-Greedy policy optimized traffic successfully!")
    else:
        print("Result: Random performed better or same (Expected in short runs).")
    print("="*40)
