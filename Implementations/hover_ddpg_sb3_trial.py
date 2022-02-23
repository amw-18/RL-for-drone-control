import numpy as np
from stable_baselines3 import DDPG 
from HoverEnvs import *
from gym_pybullet_drones.utils.utils import sync
import time


SIM_FREQ_HZ = 240
AGGR_PHY_STEPS = 1 # 1 physics step per action/control command

INIT_XYZS = np.array([[0., 0., 1.]])
INIT_RPYS = np.array([[0., 0., 0.]])
env = HoverAviary(drone_model=DroneModel.CF2X,
                initial_xyzs=INIT_XYZS,
                initial_rpys=INIT_RPYS,
                freq=SIM_FREQ_HZ,
                aggregate_phy_steps=AGGR_PHY_STEPS,
                gui=True,
                record=False,
                )

model = DDPG.load("Implementations/hover_sb3_logs/DDPG_HoverR3D1.zip", env=env)

obs = env.reset()
start = time.time()
for i in range(5*env.SIM_FREQ):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    if i%env.SIM_FREQ == 0:
        env.render()
        print(done)
    sync(i, start, env.TIMESTEP)
    if done:
        obs = env.reset()
env.close()