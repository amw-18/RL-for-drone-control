import gym_pybullet_drones
from gym_pybullet_drones.envs.BaseAviary import DroneModel
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from ray.rllib.policy import Policy
import numpy as np


class DSLPIDPolicy(Policy):
    def __init__(self, observation_space, action_space, config, drone_model: DroneModel = DroneModel.CF2X):
        Policy.__init__(self, observation_space, action_space, config)

        self.controller = DSLPIDControl(drone_model)
        self.TSTEP = 0.004166666666666667  # 1/240
        self.MAX_RPM = 21702

    def compute_actions(self, 
                        obs_batch, 
                        state_batches, 
                        prev_action_batch=None, 
                        prev_reward_batch=None, 
                        info_batch=None, 
                        episodes=None, 
                        **kwargs):
        actions = []
        for obs in obs_batch:
            # print(obs)
            state = obs[:20]
            target_rpys = obs[20:23]*np.pi
            rpms, pos_e, yaw_e = self.controller.computeControlFromState(control_timestep=self.TSTEP,
                                                                        state=state,
                                                                        target_pos=state[:3],
                                                                        target_rpy=target_rpys
                                                                        )
            rpms = 2*rpms/self.MAX_RPM - 1
            rpms = np.array([list(rpms)])
            actions.append(rpms)
        
        return actions, [], {}

    def learn_on_batch(self, samples):
        return {}




    