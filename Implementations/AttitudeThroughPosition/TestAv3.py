import numpy as np
from scipy.spatial.transform.rotation import Rotation as R
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary
from gym import spaces
from ModDSLPID import ModDSLPIDpos, ModDSLPIDatt

class TestAv3(BaseAviary):
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

        assert(self.DRONE_MODEL==DroneModel.CF2X)
        self.orig_P_COEFF_TOR = np.array([70000., 70000., 60000.])
        self.orig_I_COEFF_TOR = np.array([500.0, 500.0, 500.0])
        self.orig_D_COEFF_TOR = np.array([20000., 20000., 12000.])
        self.PID_pos_ctrl = ModDSLPIDpos(DroneModel.CF2X)
        self.PID_att_ctrl = ModDSLPIDatt(DroneModel.CF2X)

        self.computed_target_rpy = np.array([0., 0., 0.])

        self._wp_creator()

    def _wp_creator(self):
        self.target_rpy = np.array([*(np.random.rand(2)*2-1), 0])*np.pi/18   # -pi/18 to +pi/18
        # self.target_rpy = np.array([1., 1., 0.])*np.pi/3
        target_facing = R.from_euler('xyz', self.target_rpy).apply(np.array([0., 0., 1.]))

        self.NUM_WP = self.EPISODE_LEN_SEC*self.SIM_FREQ
        self.TARGET_VEL = np.sign(target_facing)*np.linspace([0, 0, 0], np.abs(target_facing)*self.REQD_SPEED, self.NUM_WP)
        self.TARGET_POS = np.zeros((self.NUM_WP,3))
        for i in range(1, self.NUM_WP):
            self.TARGET_POS[i] = self.TARGET_POS[i-1] + self.TARGET_VEL[i-1]*(1/self.SIM_FREQ)
        
        self.wp_counter = 0

    def reset(self):
        self.PID_pos_ctrl = ModDSLPIDpos(DroneModel.CF2X)
        self.PID_att_ctrl = ModDSLPIDatt(DroneModel.CF2X)

        self.computed_target_rpy = np.array([0., 0., 0.])

        self._wp_creator()
        return super().reset()

    def _actionSpace(self):
        act_lower_bound = np.array([0]*9)
        act_upper_bound = np.array([np.inf]*9)
        return spaces.Box(low  = act_lower_bound,
                          high = act_upper_bound,
                          dtype= np.float32)

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.
        """
        P_COEFF_ATT = action[:3]*self.orig_P_COEFF_TOR
        I_COEFF_ATT = action[3:6]*self.orig_I_COEFF_TOR
        D_COEFF_ATT = action[6:]*self.orig_D_COEFF_TOR
        state = self._getDroneStateVector(0)
        self.PID_att_ctrl.setPIDCoefficients(p_coeff_att=P_COEFF_ATT,
                                            i_coeff_att=I_COEFF_ATT,
                                            d_coeff_att=D_COEFF_ATT)
        rpms = self.PID_att_ctrl.computeControl(control_timestep=1/self.SIM_FREQ,
                                                state=state,
                                                thrust=self.target_z_thrust,
                                                computed_target_rpy=self.computed_target_rpy)
        return np.clip(rpms, 0, self.MAX_RPM)

    def _observationSpace(self):
        # of length 9
        #### Observation vector ### R       P       Y        WX       WY       WZ       TargetR    TargetP    TargetY
        obs_lower_bound = np.array([-1, -1, -1,  -1, -1, -1,  -1,    -1,    -1])
        obs_upper_bound = np.array([1,  1,  1,   1,  1,  1,   1,     1,     1])
        return spaces.Box(low  = obs_lower_bound,
                          high = obs_upper_bound,
                          dtype= np.float32)

    def step(self, action):
        return super().step(action)


    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (9,).

        """
        state = self._getDroneStateVector(0)
        target_pos = self.TARGET_POS[self.wp_counter]
        target_vel = self.TARGET_VEL[self.wp_counter]
        self.last_computed_target_rpy = self.computed_target_rpy

        self.target_z_thrust, self.computed_target_rpy = self.PID_pos_ctrl.computeControl(control_timestep=1/self.SIM_FREQ,
                                                                                                    state=state,
                                                                                                    target_pos=target_pos,
                                                                                                    target_vel=target_vel)
        self.wp_counter += 1
        self.wp_counter = min(self.wp_counter, self.NUM_WP-1)

        norm_rpy = state[7:10]/np.pi
        rpy_rates = state[13:16]
        if np.linalg.norm(rpy_rates) > 0:
            norm_rpy_rates = rpy_rates/np.linalg.norm(rpy_rates)
        else:
            norm_rpy_rates = rpy_rates
        norm_target_rpy = self.computed_target_rpy/np.pi
        obs = np.hstack([norm_rpy,
                         norm_rpy_rates,
                         norm_target_rpy]).reshape(9,)
        
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
        rpy_err = np.abs(self.last_computed_target_rpy - curr_rpy)
        
        return -np.sum(rpy_err)

    def _computeInfo(self):
        state = self._getDroneStateVector(0)
        curr_rpy = state[7:10]
        # Calculating rpy error
        rpy_err = np.abs(self.last_computed_target_rpy - curr_rpy)
        return {'rpy_err':rpy_err, 'target_rpy':self.last_computed_target_rpy, 'curr_rpy':curr_rpy}