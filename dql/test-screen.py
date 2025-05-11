import gymnasium as gym
import time

game = "CartPole-v1"
#game = "MsPacman-v4"
try:
    env = gym.make(game, render_mode="human")
    observation, info = env.reset()
    print("CartPole environment created with human rendering.")
    for _ in range(200):
        action = env.action_space.sample()  # take a random action
        observation, reward, terminated, truncated, info = env.step(action)
        env.render() # Explicitly call render, though step should also trigger it
        time.sleep(0.05)
        if terminated or truncated:
            print("Episode finished.")
            observation, info = env.reset()
    env.close()
    print("CartPole test finished.")
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()