import numpy as np 
from gym import spaces
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary

class AttitudeTestAviary(BaseAviary):
    def __init__(self,
                 physics: Physics=Physics.PYB,
                 gui=False,
                 record=False
                 ):
        
        self.EPISODE_LEN_SEC = 5

        super().__init__(drone_model=DroneModel.CF2X,
                        num_drones=1,
                        neighbourhood_radius=np.inf,
                        initial_xyzs=np.array([[0., 0., 1.]]),
                        initial_rpys=np.array([[0., 0., 0.]]),
                        physics=physics,
                        freq=240,
                        aggregate_phy_steps=1,
                        gui=gui,
                        record=record,
                        obstacles=False,
                        user_debug_gui=False,
                        vision_attributes=False,
                        dynamics_attributes=True)

        rp_limits = np.pi/18
        final_target = np.array([rp_limits, rp_limits, 0])
        self.targets = [i*final_target/self.EPISODE_LEN_SEC \
                        for i in range(1, self.EPISODE_LEN_SEC+1)]
        self.target_rpys = self.targets[0]

    def reset(self):
        self.target_rpys = self.targets[0]
        return super().reset()

    def _actionSpace(self):
        """
        Thrust and Torque action: [P1, P2, P3, P4] 
        """
        return spaces.Box(low  =np.array([-1., -1., -1., -1.]),
                          high =np.array([ 1.,  1.,  1.,  1.]),
                          dtype=np.float32)

    def _preprocessAction(self, action):
        """
        Preprocessing the agent's action to motor rpms
        """
        # print('here', action)
        return self._normalizedActionToRPM(action)

    def _observationSpace(self):
        """
        OBS OF SIZE 20 (WITH QUATERNION AND RPMS)
        """
        #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ       P0            P1            P2            P3
        obs_lower_bound = np.array([-np.inf, -np.inf, 0.,     -1., -1., -1., -1., -np.pi, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0.,           0.,           0.,           0.])
        obs_upper_bound = np.array([np.inf,  np.inf,  np.inf, 1.,  1.,  1.,  1.,  np.pi,  np.pi,  np.pi,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM])          
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (20,).

        """
        obs = self._getDroneStateVector(0)
        return obs

    def _computeDone(self):
        if self.step_counter*self.TIMESTEP >= self.EPISODE_LEN_SEC:
            return True
        else:
            self.target_rpys = self.targets[int(self.step_counter*self.TIMESTEP)]
            return False

    def _computeReward(self):
        # Getting current rpy values
        state = self._getDroneStateVector(0)
        curr_rpy = state[7:10]
        # Calculating rpy error
        rpy_err = self.target_rpys - curr_rpy
        # Normalizing
        rpy_err = np.abs(rpy_err / np.pi)
        # The higher the sum of errors, the lower the reward
        return -np.sum(rpy_err)

    def _computeInfo(self):
        state = self._getDroneStateVector(0)
        curr_rpy = state[7:10]
        # Calculating rpy error
        rpy_err = self.target_rpys - curr_rpy
        return {'rpy_err': rpy_err}