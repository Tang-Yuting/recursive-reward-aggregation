import os

from recursive_stable_baselines3.recursive_ppo import Recursive_PPO, Recursive_PPO_multi_output
from recursive_stable_baselines3.recursive_td3 import TD3

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_file) as file_handler:
    __version__ = file_handler.read().strip()


__all__ = [
    "Recursive_PPO",
    "Recursive_PPO_multi_output",
    "TD3",
]
