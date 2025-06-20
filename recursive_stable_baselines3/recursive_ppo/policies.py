# This file is here just to define MlpPolicy/CnnPolicy
# that work for PPO
from recursive_stable_baselines3.recursive_common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, ActorCriticPolicy_multi_output, MultiInputActorCriticPolicy

MlpPolicy = ActorCriticPolicy
MlpPolicy_multi_output = ActorCriticPolicy_multi_output
CnnPolicy = ActorCriticCnnPolicy
MultiInputPolicy = MultiInputActorCriticPolicy
