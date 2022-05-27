import numpy as np
from gym import spaces
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

class AdaptivePIDAviary(BaseAviary):
    def __init__(self,
                 physics: Physics=Physics.PYB,
                 gui=False,
                 record=False
                 ):
        
        self.EPISODE_LEN_SEC = 1
        self.CTRL_FREQ = 240
        self.control_timestep = 1/self.CTRL_FREQ

        super().__init__(drone_model=DroneModel.CF2X,
                        num_drones=1,
                        neighbourhood_radius=np.inf,
                        initial_xyzs=np.array([[0., 0., 1.]]),
                        initial_rpys=np.array([[0., 0., 0.]]),
                        physics=physics,
                        freq=self.CTRL_FREQ,
                        aggregate_phy_steps=1,
                        gui=gui,
                        record=record,
                        obstacles=False,
                        user_debug_gui=False,
                        vision_attributes=False,
                        dynamics_attributes=True)

        self.controller = DSLPIDControl(DroneModel.CF2X)

        self.reset()

    def reset(self):
        self.target_rpys = self._create_target_rpy()
        self.controller.reset()
        return super().reset()
    
    def _create_target_rpy(self):
        target_rp = (np.random.rand(2)*2-1)*np.pi/36   # -pi/36 to +pi/36
        # target_y = (np.random.rand(2)*2-1)*np.pi      # -pi to +pi
        target_y = [0]
        return np.array([target_rp[0], target_rp[1], target_y[0]])

    def _actionSpace(self):
        """
        PID values : 9 in number for attitude related PID values
        (going from 0 to (2 times + 10% of max) the nominal amount)
        """
        P_COEFF_TOR = 2*np.array([70000., 70000., 60000.]) + 0.1*max([70000., 70000., 60000.])
        I_COEFF_TOR = 2*np.array([.0, .0, 500.]) + 0.1*max([.0, .0, 500.])
        D_COEFF_TOR = 2*np.array([20000., 20000., 12000.]) + 0.1*max([20000., 20000., 12000.])
        return spaces.Box(low  = np.zeros((9,)),
                          high = np.hstack([P_COEFF_TOR,
                                           I_COEFF_TOR,
                                           D_COEFF_TOR]).reshape((9,)),
                          dtype=np.float32)

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.
        """
        
        self.controller.setPIDCoefficients(p_coeff_att=action[:3],
                                           i_coeff_att=action[3:6],
                                           d_coeff_att=action[6:9])

        state = self._getDroneStateVector(0)

        rpms = self.controller._dslPIDAttitudeControl(self.control_timestep,
                                                    thrust=0,
                                                    cur_quat=state[3:7],
                                                    target_euler=self.target_rpys,
                                                    target_rpy_rates=np.zeros(3)
                                                    )
        rpms = np.clip(rpms, 0, self.MAX_RPM)
        return rpms

    def _observationSpace(self):
        """
        OBS OF SIZE 9
        Observation vector -------- R      P      Y           WX      WY      WZ            TR,    TP,    TY
        """
        obs_lower_bound = np.array([-np.pi,-np.pi,-np.pi,     -np.inf,-np.inf,-np.inf,      -np.pi,-np.pi,-np.pi])
        obs_upper_bound = np.array([ np.pi, np.pi, np.pi,      np.inf, np.inf, np.inf,       np.pi, np.pi, np.pi])          
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (9,).

        """
        obs = self._clipAndNormalizeState(self._getDroneStateVector(0))
        
        # Adding target rpys to the observation 
        obs = np.hstack([obs, 
                        self.target_rpys/np.pi]).reshape((9,))

        return obs

    def _clipAndNormalizeState(self,
                               state
                               ):
        """Normalizes a drone's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (6,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (6,)-shaped array of floats containing the normalized state of a single drone.

        """

        MAX_PITCH_ROLL = np.pi # Full range

        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)

        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi # No reason to clip
        normalized_ang_vel = state[13:16]/np.linalg.norm(state[13:16]) if np.linalg.norm(state[13:16]) != 0 else state[13:16]

        norm_and_clipped = np.hstack([normalized_rp,
                                      normalized_y,
                                      normalized_ang_vel,
                                      ]).reshape(6,)

        return norm_and_clipped

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

        

    

