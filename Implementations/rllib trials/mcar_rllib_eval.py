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

checkpoint_path = "C:/Users/awals/ray_results/DDPG_mcar_env_2022-02-17_19-02-490gw6k_9a/checkpoint_000015/checkpoint-15"

trainer.restore(checkpoint_path)

print(trainer.evaluate()["evaluation"])

# env = gym.make("MountainCarContinuous-v0")
# obs = env.reset()
# done = False
# total_reward = 0.0
# while not done:
#     action = trainer.compute_single_action(observation=obs)
#     obs, reward, done, info = env.step(action)
#     total_reward += reward

# print(f"Total reward for 1 episode = {total_reward}")

# trainer.get_policy().compute_single_action(observ)