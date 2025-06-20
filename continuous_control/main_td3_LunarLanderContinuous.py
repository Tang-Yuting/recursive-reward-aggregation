import sys
import argparse
import os

# sys.path.insert(0, "/workspace/stable-baselines3")
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))

import gymnasium as gym
from recursive_stable_baselines3 import TD3

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
import numpy as np

parser = argparse.ArgumentParser(description="Train PPO on Hopper-v4")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--env", type=str, default="Hopper-v4", help="Gym environment name")
parser.add_argument("--env_name", type=str, default="Hopper", help="Gym environment name for file")
parser.add_argument("--recursive_type", type=str, default="min", help="Recursive type")
parser.add_argument("--output_number", type=int, default=1, help="output number")
args = parser.parse_args()

seed = args.seed
set_random_seed(seed)

env = gym.make(args.env)
obs, _ = env.reset(seed=args.seed)
env.action_space.seed(seed)
env = Monitor(env)

model = TD3(
    "MlpPolicy",
    env,
    verbose=1,
    seed=args.seed,
    learning_rate=3e-4,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    train_freq=(1, "episode"),
    gradient_steps=100,
    policy_delay=2,
    target_policy_noise=0.2,
    target_noise_clip=0.5,
    buffer_size=1000000,
    learning_starts=10000,
    recursive_type=args.recursive_type,
    output_number=args.output_number,
)


model.learn(total_timesteps=1000000)
model.save(f"result_TD3/{args.env_name}/{args.recursive_type}/{args.seed}/TD3_model_{args.recursive_type}_{args.seed}")

import numpy as np

num_episodes = 100
success_count = 0
fail_count = 0
total_rewards = []
velocities = []
angles = []
fuel_usage = []
landing_times = []

for _ in range(num_episodes):
    obs, _ = env.reset()
    done = False
    truncated = False
    episode_reward = 0
    step_count = 0
    max_vx, max_vy = 0, 0
    total_fuel = 0
    total_angle_deviation = 0

    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        step_count += 1

        x, y, vx, vy, theta, vtheta, left_leg, right_leg = obs

        max_vx = max(max_vx, abs(vx))
        max_vy = max(max_vy, abs(vy))
        total_angle_deviation += abs(theta)
        main_thrust = np.clip(action[0], 0, 1)
        side_thrust = np.clip(action[1], -1, 1)
        total_fuel += 0.3 * main_thrust + 0.03 * abs(side_thrust)

    if reward >= 100:
        success_count += 1
    else:
        fail_count += 1

    total_rewards.append(episode_reward)
    velocities.append((max_vx, max_vy))
    angles.append(total_angle_deviation / step_count)
    fuel_usage.append(total_fuel)
    landing_times.append(step_count)

success_rate = success_count / num_episodes
avg_reward = np.mean(total_rewards)
worst_reward = np.min(total_rewards)
avg_velocity = np.mean(velocities, axis=0)
avg_angle_stability = np.mean(angles)
avg_fuel_usage = np.mean(fuel_usage)
avg_landing_time = np.mean(landing_times)

print("===== Evaluation Summary =====")
print(f"Success Rate: {success_rate:.2%}")
print(f"Avg Reward: {avg_reward:.2f}, Worst Reward: {worst_reward:.2f}")
print(f"Avg Max Velocity: vx = {avg_velocity[0]:.2f}, vy = {avg_velocity[1]:.2f}")
print(f"Avg Angle Deviation: {avg_angle_stability:.2f} rad")
print(f"Avg Fuel Usage: {avg_fuel_usage:.2f}")
print(f"Avg Landing Time: {avg_landing_time:.2f} steps")

env.close()
