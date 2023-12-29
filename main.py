import gymnasium as gym
from matplotlib import pyplot as plt
#from stable_baselines3 import DQN
#import time

env = gym.make('merge-v0', render_mode='rgb_array')
env.reset()
for i in range(100):
    action = env.action_type.actions_indexes["IDLE"]
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    print(i)

    # Introduce a delay of 0.1 seconds (adjust as needed)
    #time.sleep(0.001)

plt.imshow(env.render())
plt.show()
