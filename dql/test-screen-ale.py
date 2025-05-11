import gymnasium
import ale_py

gymnasium.register_envs(ale_py)

env = gymnasium.make("MsPacman-v4", render_mode="human")
env.reset()
for _ in range(200):
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()