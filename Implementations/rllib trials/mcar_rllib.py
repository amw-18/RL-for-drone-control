import gym
from ray.tune.registry import register_env
from ray.rllib.agents.ddpg import DDPGTrainer
from ray.rllib.agents.ddpg import DEFAULT_CONFIG
import yaml

def mcar_env(env_config):
    return gym.make("MountainCarContinuous-v0")

register_env("mcar_env", mcar_env)

configfile = "rllib trials/mountaincarcontinuous-ddpg.yaml"
with open(configfile) as f:
    my_file = yaml.safe_load(f)

trainer = DDPGTrainer(config=my_file["config"])

for i in range(20):
    results = trainer.train()
    print(f"Iter: {i}, avg. reward={results['episode_reward_mean']}")

# Evaluation
env = mcar_env({})
obs = env.reset()
done = False
total_reward = 0.0
while not done:
    action = trainer.compute_single_action(obs)
    obs, reward, done, info = env.step(action)
    total_reward += reward

print(f"Total reward for 1 episode = {total_reward}")

