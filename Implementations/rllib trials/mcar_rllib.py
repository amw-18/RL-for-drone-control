import gym
from ray.tune.registry import register_env
from ray.rllib.agents.ddpg import DDPGTrainer
from ray.rllib.agents.ddpg import DEFAULT_CONFIG
import yaml

register_env("mcar_env", lambda _: gym.make("MountainCarContinuous-v0"))

# Loading preset hyperparameter values from a yaml file
configfile = "Implementations/rllib trials/mountaincarcontinuous-ddpg.yaml"
with open(configfile) as f:
    my_file = yaml.safe_load(f)

# Creating the trainer
trainer = DDPGTrainer(config=my_file["config"])

# Training for 'num_iter' iterations
num_iter = 2
for i in range(num_iter):
    result = trainer.train()
    print(f"Iter: {i}, avg. reward={result['episode_reward_mean']}")

trainer.save()

# Evaluation for a single episode
env = gym.make("MountainCarContinuous-v0")
obs = env.reset()
done = False
total_reward = 0.0
while not done:
    action = trainer.compute_single_action(obs)
    obs, reward, done, info = env.step(action)
    total_reward += reward

print(f"Total reward for 1 episode = {total_reward}")