import sys
import argparse
import os

# sys.path.insert(0, "/workspace/stable-baselines3")
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))

from recursive_stable_baselines3 import TD3

import gymnasium as gym
import numpy as np

from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor

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

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = TD3(
    "MlpPolicy",
    env,
    verbose=1,
    seed=args.seed,
    learning_rate=3e-4,
    batch_size=512,
    gamma=0.99,
    tau=0.005,
    train_freq=(200, "step"),
    gradient_steps=100,
    policy_delay=2,
    target_policy_noise=0.2,
    target_noise_clip=0.5,
    buffer_size=int(2e6),
    learning_starts=50000,
    action_noise=action_noise,
    recursive_type=args.recursive_type,
    output_number=args.output_number,
)

model.learn(total_timesteps=3000000)
model.save(f"result_TD3/{args.env_name}/{args.recursive_type}/{args.seed}/TD3_model_{args.recursive_type}_{args.seed}")


def evaluate_model(model, env, num_episodes=50):
    metrics = {
        "avg_reward": [],
        "max_distance": [],
        "min_distance": float("inf"),
        "avg_velocity": [],
        "avg_energy_usage": [],
        "avg_stability": []
    }

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done, truncated = False, False
        episode_reward = 0
        total_distance = 0
        total_energy_usage = 0
        total_stability = 0
        step_count = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward

            forward_velocity = obs[13]
            energy_penalty = np.sum(np.abs(action))
            stability = -np.var(obs[:3])

            total_distance += forward_velocity
            total_energy_usage += energy_penalty
            total_stability += stability
            step_count += 1

        metrics["avg_reward"].append(episode_reward)
        metrics["max_distance"].append(total_distance)
        metrics["min_distance"] = min(metrics["min_distance"], total_distance)
        metrics["avg_velocity"].append(total_distance / step_count)
        metrics["avg_energy_usage"].append(total_energy_usage / step_count)
        metrics["avg_stability"].append(total_stability / step_count)

    metrics["avg_reward"] = np.mean(metrics["avg_reward"])
    metrics["avg_velocity"] = np.mean(metrics["avg_velocity"])
    metrics["avg_energy_usage"] = np.mean(metrics["avg_energy_usage"])
    metrics["avg_stability"] = np.mean(metrics["avg_stability"])
    metrics["max_distance"] = max(metrics["max_distance"])
    metrics["min_distance"] = metrics["min_distance"]

    print("\n===== Evaluation Summary =====")
    print(f"Avg Reward: {metrics['avg_reward']:.2f}")
    print(f"Avg Velocity: {metrics['avg_velocity']:.2f} m/s")
    print(f"Avg Energy Usage: {metrics['avg_energy_usage']:.2f}")
    print(f"Max Distance: {metrics['max_distance']:.2f} m")
    print(f"Min Distance: {metrics['min_distance']:.2f} m")
    print(f"Avg Stability: {metrics['avg_stability']:.4f}")

    return metrics

evaluate_model(model, env)
env.close()

