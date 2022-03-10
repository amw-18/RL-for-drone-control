import numpy as np 
import ray
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo import DEFAULT_CONFIG
import yaml
import time
from gym_pybullet_drones.utils.utils import sync
from HoverEnvs import *
import matplotlib.pyplot as plt 

if __name__ == "__main__":
    SIM_FREQ_HZ = 240
    AGGR_PHY_STEPS = 1 # 1 physics step per action/control command

    INIT_XYZS = np.array([[0., 0., 1.]])
    INIT_RPYS = np.array([[0., 0., 0.]])

    register_env("HoverR1D1", lambda _: HoverR1D1(drone_model=DroneModel.CF2X,
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
    config["env"] = "HoverR1D1"
    config["vf_clip_param"] = 1000
    config["lambda"] = 0.95
    config["rollout_fragment_length"] = 1000
    config["lr"] = 3e-4
    
    
    trainer = PPOTrainer(config=config)

    for i in range(250):  # 1M timesteps. Expected time 1.9 hours with 10 workers
        results = trainer.train()
        print(f"Iter: {i}, {results['timesteps_total']} timesteps total.")

    policy = trainer.get_policy()
    trainer.save()
    ray.shutdown()


    # Evaluation
    env = HoverAviary(drone_model=DroneModel.CF2X,
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