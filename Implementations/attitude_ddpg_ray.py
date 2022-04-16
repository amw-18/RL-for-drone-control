import numpy as np 
import ray
from ray.tune.registry import register_env
from ray.rllib.agents.ddpg import DDPGTrainer
from ray.rllib.agents.ddpg import DEFAULT_CONFIG
import yaml
import time
from gym_pybullet_drones.utils.utils import sync
from AttitudeAviary import *
import matplotlib.pyplot as plt 

if __name__ == "__main__":
    SIM_FREQ_HZ = 1000
    AGGR_PHY_STEPS = 1 # 1 physics step per action/control command

    register_env("AttitudeAviary1_1", lambda _: AttitudeAviary1_1())

    config = DEFAULT_CONFIG.copy()
    config["num_workers"] = 10
    config["framework"] = "torch"
    config["env"] = "AttitudeAviary1_1"
    
    trainer = DDPGTrainer(config=config)

    for i in range(1000):  # 1000 iters -> 1M timesteps. Expected time 1.9 hours with 10 workers
        results = trainer.train()
        print(f"Iter: {i}, {results['timesteps_total']} timesteps total.")

    policy = trainer.get_policy()
    trainer.save()
    ray.shutdown()


    # Evaluation
    env = AttitudeAviary1_1(record=True)

    obs = env.reset()
    start = time.time()
    rew = []
    for i in range(1*env.SIM_FREQ):
        action, _states, _dict = policy.compute_single_action(obs)
        obs, reward, done, info = env.step(action)
        rew.append(reward)
        if i%env.SIM_FREQ == 0:
            env.render()
            print(done)
        sync(i, start, env.TIMESTEP)
        if done:
            obs = env.reset()
    env.close()

    plt.plot(rew)
    plt.show()