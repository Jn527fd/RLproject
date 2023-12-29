import highway_env

class MergeLaneEgoEnvironment:
    def __init__(self):
        # Create the highway_env environment with the ego vehicle starting on the merge lane
        config = {
            "lanes_count": 3,            # Adjust the number of lanes as needed
            "vehicles_count": 10,        # Adjust the number of vehicles as needed
            "initial_lane_id": 1,        # Specify the merge lane as the initial lane for the ego vehicle
            "duration": 40,              # Adjust the simulation duration as needed
            "ego_spacing": 2.5           # Adjust the initial spacing for the ego vehicle
        }
        self.env = highway_env.make("merge-v0", config_dict=config)

    def reset(self):
        # Reset the environment to the initial state
        obs = self.env.reset()
        return obs

    def step(self, action):
        # Take a step in the environment based on the given action
        # Return the next state, reward, done flag, and additional information
        next_state, reward, done, info = self.env.step(action)
        return next_state, reward, done, info
    
    def default_policy(self, state):
        # Placeholder for a simple default policy
        # Replace this with your actual policy logic based on the state information
        # For example, a simple policy might be to maintain the current speed
        return {"acceleration": 1.0, "steering": 0.0}

# Example usage
merge_lane_ego_env = MergeLaneEgoEnvironment()
initial_state = merge_lane_ego_env.reset()

# Run the environment for some steps
for _ in range(100):
    action = merge_lane_ego_env.default_policy(initial_state)  # Replace with your actual policy function
    next_state, reward, done, info = merge_lane_ego_env.step(action)
    initial_state = next_state

# Close the environment when done
merge_lane_ego_env.env.close()
