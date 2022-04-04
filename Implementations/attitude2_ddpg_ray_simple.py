import numpy as np 
import ray
from ray.tune.registry import register_env
from ray.rllib.agents.ddpg.ddpg_torch_policy import DDPGTorchPolicy
from ray.rllib.agents.ddpg import DEFAULT_CONFIG
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
import yaml
import time
from gym_pybullet_drones.utils.utils import sync
from AttitudeAviary2 import *
import matplotlib.pyplot as plt 


if __name__ == "__main__":
    n_episodes = 10

    env = AttitudeAviary2(gui=True)

    policy = DDPGTorchPolicy(env.observation_space, env.action_space, DEFAULT_CONFIG)

    worker = RolloutWorker(env_creator=AttitudeAviary2,
                            policy_spec=DDPGTorchPolicy,
                            batch_mode="complete_episodes")

    for _ in range(n_episodes):
        weights = {DEFAULT_POLICY_ID:policy.get_weights()}
        worker.set_weights(weights)
        batch = worker.sample()
        policy.learn_on_batch(batch)
        

    total_reward = 0.0
    obs = env.reset()
    done = False
    STEP = 0
    START = time.time()
    while not done:
        action = policy.compute_single_action(obs)
        obs, reward, done, info = env.step(action[0])
        total_reward += reward
        STEP += 1
        env.render()
        sync(START, STEP, env.TIMESTEP)

    print(total_reward)
    env.close()