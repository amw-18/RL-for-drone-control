import numpy as np 
import ray
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo import DEFAULT_CONFIG
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
import yaml
import time
from gym_pybullet_drones.utils.utils import sync
import matplotlib.pyplot as plt
from AttitudeAviary2 import *


def custom_eval(env, policy):
    rpy_errs = []
    cumm_reward = 0
    obs = env.reset()
    start = time.time()
    for i in range(1*env.SIM_FREQ):
        action, _states, _dict = policy.compute_single_action(obs)
        obs, reward, done, info = env.step(action)
        cumm_reward += reward
        rpy_errs.append(info['rpy_err'])
        if env.GUI:
            if i%env.SIM_FREQ == 0:
                env.render()
                print(done)
            sync(i, start, env.TIMESTEP)
            if done:
                break

    return cumm_reward, rpy_errs


if __name__ == "__main__":
    env_name = "AttitudeAviary2_3"

    register_env(env_name, lambda _: AttitudeAviary2_3())


    config = {}
    config["num_workers"] = 0
    config["framework"] = "torch"
    
    config["env"] = env_name

    config["sgd_minibatch_size"] = 64
    config["num_sgd_iter"] = 10
    config["lr"] = 3e-4
    config["lambda"] = 0.95
    config["clip_param"] = 0.2

    
    eval_env = AttitudeAviary2_3(gui=False)
    trainer = PPOTrainer(config=config)

    eval_history = []
    for iter in range(1000):
        results = trainer.train()
        if iter % 5 == 0:
            eval_history.append(custom_eval(eval_env, trainer.get_policy(DEFAULT_POLICY_ID)))
            print(f"Iter: {iter} Evaluation Episode Reward: {eval_history[-1][0]}")

    trainer.save()
    ray.shutdown()
    
    plt.figure()
    rew_history = [x[0] for x in eval_history]
    plt.plot(rew_history)
    plt.xlabel("Episodes (x5)")
    plt.ylabel("Cummulative Reward per Episode")
    plt.savefig("Implementations/Plots/rew_history_2_3_1.png")

    plt.figure()
    rpy_curve = eval_history[-1][1]
    plt.plot(rpy_curve)
    plt.xlabel("Timestep")
    plt.ylabel("RPY_error")
    plt.savefig("Implementations/Plots/final_rpy_curve_2_3_1.png")


    plt.show()