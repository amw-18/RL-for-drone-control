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



env_name = "AttitudeAviary2_2"

register_env(env_name, lambda _: AttitudeAviary2_2())

checkpoint_path = "Implementations/SavedTrainers/PPO_AttitudeAviary2_2_2022-04-26_10-48-51_4wh8i2q/checkpoint_001000/checkpoint-1000"
config = {}
config["num_workers"] = 0
config["framework"] = "torch"

config["env"] = env_name

config["sgd_minibatch_size"] = 64
config["num_sgd_iter"] = 10
config["lr"] = 3e-4
config["lambda"] = 0.95
config["clip_param"] = 0.2

trainer = PPOTrainer(config=config)

trainer.restore(checkpoint_path)

# trainer.evaluate()

eval_env = AttitudeAviary2_2(gui=False)

cumm_reward, rpy_errs = custom_eval(eval_env, trainer.get_policy())

roll_errs = np.array([v[0] for v in rpy_errs])
pitch_errs = np.array([v[1] for v in rpy_errs])
yaw_errs = np.array([v[2] for v in rpy_errs])

print(cumm_reward)
plt.plot(roll_errs/roll_errs[0])
plt.plot(pitch_errs/pitch_errs[0])
plt.xlabel("Timestep")
plt.ylabel("RPY_error")
plt.legend(["roll-error", "pitch-error"])
plt.show()
