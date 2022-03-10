import numpy as np 
import ray
from ray.tune.registry import register_env
from ray.rllib.agents.ddpg import DDPGTrainer
from ray.rllib.agents.ddpg import DEFAULT_CONFIG
import yaml
import time
from gym_pybullet_drones.utils.utils import sync
from AltHoldEnvs import *
import matplotlib.pyplot as plt 

if __name__ == "__main__":
    SIM_FREQ_HZ = 240
    AGGR_PHY_STEPS = 1 # 1 physics step per action/control command

    INIT_XYZS = np.array([[0., 0., 1.]])
    INIT_RPYS = np.array([[0., 0., 0.]])

    register_env("AltHoldAviary", lambda _: AltHoldAviary(drone_model=DroneModel.CF2X,
                                                    initial_xyzs=INIT_XYZS,
                                                    initial_rpys=INIT_RPYS,
                                                    freq=SIM_FREQ_HZ,
                                                    aggregate_phy_steps=AGGR_PHY_STEPS,
                                                    gui=False,
                                                    record=False,
                                                    )
                    )

    config = DEFAULT_CONFIG.copy()
    config["num_workers"] = 10
    config["framework"] = "torch"
    config["env"] = "AltHoldAviary"
    
    trainer = DDPGTrainer(config=config)

    for i in range(1000):  # 1M timesteps. Expected time 1.9 hours with 10 workers
        results = trainer.train()
        print(f"Iter: {i}, {results['timesteps_total']} timesteps total.")

    policy = trainer.get_policy()
    trainer.save()
    ray.shutdown()


    # Evaluation
    env = AltHoldAviary(drone_model=DroneModel.CF2X,
                        initial_xyzs=INIT_XYZS,
                        initial_rpys=INIT_RPYS,
                        freq=SIM_FREQ_HZ,
                        aggregate_phy_steps=AGGR_PHY_STEPS,
                        gui=False,
                        record=True,
                        )

    obs = env.reset()
    start = time.time()
    for i in range(5*env.SIM_FREQ):
        action, _states, _dict = policy.compute_single_action(obs)
        obs, reward, done, info = env.step(action)
        if i%env.SIM_FREQ == 0:
            env.render()
            print(done)
        sync(i, start, env.TIMESTEP)
        if done:
            obs = env.reset()
    env.close()