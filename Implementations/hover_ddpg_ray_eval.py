import numpy as np 
import ray
from ray.tune.registry import register_env
from ray.rllib.agents.ddpg import DDPGTrainer
from ray.rllib.agents.ddpg import DEFAULT_CONFIG
from stable_baselines3 import DDPG
import yaml
import time
from gym_pybullet_drones.utils.utils import sync
from HoverEnvs import *


SIM_FREQ_HZ = 240
AGGR_PHY_STEPS = 1 # 1 physics step per action/control command

INIT_XYZS = np.array([[0., 0., 1.]])
INIT_RPYS = np.array([[0., 0., 0.]])

register_env("HoverR2D2", lambda _: HoverR2D2(drone_model=DroneModel.CF2X,
                                                    initial_xyzs=INIT_XYZS,
                                                    initial_rpys=INIT_RPYS,
                                                    freq=SIM_FREQ_HZ,
                                                    aggregate_phy_steps=AGGR_PHY_STEPS,
                                                    gui=False,
                                                    record=True,
                                                    )
)

checkpoint_path = "C:/Users/awals/ray_results/DDPG_HoverR2D2_2022-02-10_17-20-137e2h1pxx/checkpoint_000200/checkpoint-200"

config = DEFAULT_CONFIG.copy()
config["num_workers"] = 0
config["framework"] = "torch"
config["env"] = "HoverR2D2"
config["evaluation_num_episodes"] = 1
trainer = DDPGTrainer(config=config)

trainer.restore(checkpoint_path)

trainer.evaluate()

# env = HoverR2D2(drone_model=DroneModel.CF2X,
#                     initial_xyzs=INIT_XYZS,
#                     initial_rpys=INIT_RPYS,
#                     freq=SIM_FREQ_HZ,
#                     aggregate_phy_steps=AGGR_PHY_STEPS,
#                     gui=False,
#                     record=True,
#                     )

# total_reward = 0.0
# obs = env.reset()
# done = False
# STEP = 0
# START = time.time()
# while not done:
#     action = trainer.get_policy().compute_single_action(obs)
#     obs, reward, done, info = env.step(action)
#     total_reward += reward
#     STEP += 1
#     env.render()
#     sync(START, STEP, env.TIMESTEP)

# print(total_reward)
# env.close()
