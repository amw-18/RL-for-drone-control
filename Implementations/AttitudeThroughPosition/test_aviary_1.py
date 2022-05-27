import numpy as np
from scipy.spatial.transform.rotation import Rotation as R
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary
from gym import spaces

class TestAviary1(BaseAviary):
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 ep_len: int=1,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=np.array([[0., 0., 0.]]),
                 initial_rpys=np.array([[0., 0., 0.]]),
                 physics: Physics=Physics.PYB,
                 freq: int=240,
                 aggregate_phy_steps: int=1,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 vision_attributes=False,
                 dynamics_attributes=False
                 ):
        super().__init__(drone_model, 
                        num_drones, 
                        neighbourhood_radius, 
                        initial_xyzs, 
                        initial_rpys, 
                        physics, 
                        freq, 
                        aggregate_phy_steps, 
                        gui, 
                        record, 
                        obstacles, 
                        user_debug_gui, 
                        vision_attributes, 
                        dynamics_attributes)
        
        self.REQD_SPEED = 1 # m/s
        self.EPISODE_LEN_SEC = ep_len
        
        self._wp_creator()

    def _wp_creator(self):
        self.target_rpy = np.array([*(np.random.rand(2)*2-1), 0])*np.pi/6   # -pi/6 to +pi/6
        target_facing = R.from_euler('xyz', self.target_rpy).apply(np.array([0., 0., 1.]))

        self.NUM_WP = self.EPISODE_LEN_SEC*self.SIM_FREQ
        self.TARGET_VEL = np.sign(target_facing)*np.linspace([0, 0, 0], np.abs(target_facing)*self.REQD_SPEED, self.NUM_WP)
        self.TARGET_POS = np.zeros((self.NUM_WP,3))
        for i in range(1, self.NUM_WP):
            self.TARGET_POS[i] = self.TARGET_POS[i-1] + self.TARGET_VEL[i-1]*(1/self.SIM_FREQ)
        
        self.wp_counter = 0

    def reset(self):
        # self._wp_creator()
        return super().reset()

    def _actionSpace(self):
        """
        Thrust and Torque action: [P1, P2, P3, P4] 
        """
        act_lower_bound = np.array([0.,           0.,           0.,           0.])
        act_upper_bound = np.array([self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM])
        return spaces.Box(low  = act_lower_bound,
                          high = act_upper_bound,
                          dtype= np.float32)

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Clips and converts a dictionary into a 2D array.

        Parameters
        ----------
        action : dict[str, ndarray]
            The (unbounded) input action for each drone, to be translated into feasible RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        clipped_action = np.clip(action, 0, self.MAX_RPM)
        return clipped_action

    def _observationSpace(self):
        #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ       P0            P1            P2            P3
        obs_lower_bound = np.array([-np.inf, -np.inf, 0.,     -1., -1., -1., -1., -np.pi, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0.,           0.,           0.,           0., -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
        obs_upper_bound = np.array([np.inf,  np.inf,  np.inf, 1.,  1.,  1.,  1.,  np.pi,  np.pi,  np.pi,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
        return spaces.Box(low  = obs_lower_bound,
                          high = obs_upper_bound,
                          dtype= np.float32)

    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (20,).

        """
        obs = np.hstack([self._getDroneStateVector(0),
                         self.TARGET_POS[self.wp_counter],
                         self.TARGET_VEL[self.wp_counter]]).reshape(26,)
        self.wp_counter += 1
        self.wp_counter = min(self.wp_counter, self.NUM_WP-1)
        return obs

    def _computeDone(self):
        if self.step_counter*self.TIMESTEP >= self.EPISODE_LEN_SEC:
            return True
        else:
            return False

    def _computeReward(self):
        # Getting current rpy values
        state = self._getDroneStateVector(0)
        curr_rpy = state[7:10]
        # Calculating rpy error
        rpy_err = self.target_rpy - curr_rpy
        # Normalizing
        rpy_err = np.abs(rpy_err / (self.target_rpy+np.pi/18*0.01))
        # The higher the sum of errors, the lower the reward
        return -np.sum(rpy_err)

    def _computeInfo(self):
        return {}