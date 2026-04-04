import random

class TrafficEnv:
    def __init__(self):
        self.state = {
            'north': 0,
            'south': 0,
            'east': 0,
            'west': 0
        }
        self.pass_rate = 5
        
    def reset(self):
        """Initializes the environment with random car counts."""
        self.state = {
            'north': random.randint(5, 15),
            'south': random.randint(5, 15),
            'east': random.randint(5, 15),
            'west': random.randint(5, 15)
        }
        return self.state.copy()
        
    def step(self, action):
        """
        Executes one step in the environment.
        action = 0: North-South green
        action = 1: East-West green
        """
        # Decrease cars for the green signal directions
        if action == 0:
            self.state['north'] = max(0, self.state['north'] - self.pass_rate)
            self.state['south'] = max(0, self.state['south'] - self.pass_rate)
        elif action == 1:
            self.state['east'] = max(0, self.state['east'] - self.pass_rate)
            self.state['west'] = max(0, self.state['west'] - self.pass_rate)
            
        # Add incoming cars to all directions
        for direction in self.state:
            self.state[direction] += random.randint(0, 3)
            
        return self.state.copy()

if __name__ == "__main__":
    env = TrafficEnv()
    state = env.reset()
    
    print("Initial State:", state)
    print("-" * 30)
    
    for i in range(1, 11): # Run for 10 steps
        action = random.choice([0, 1])
        
        signal = "North-South Green" if action == 0 else "East-West Green"
        
        state = env.step(action)
        
        print(f"Step: {i}")
        print(f"Signal: {signal}")
        print(f"State: {state}")
        print("-" * 30)
