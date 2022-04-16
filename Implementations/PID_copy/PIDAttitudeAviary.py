import numpy as np 
from gym import spaces
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary


class PIDAttitudeAviary(BaseAviary):
    def __init__(self,
                 physics: Physics=Physics.PYB,
                 gui=False,
                 record=False
                 ):
        
        self.EPISODE_LEN_SEC = 1

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

        self.target_rpys = self._create_target_rpy()

    def reset(self):
        self.target_rpys = self._create_target_rpy()
        return super().reset()
    
    def _create_target_rpy(self):
        target_rp = (np.random.rand(2)*2-1)*np.pi/18   # -pi/18 to +pi/18
        # target_y = (np.random.rand(2)*2-1)*np.pi      # -pi to +pi
        target_y = [0]
        return np.array([target_rp[0], target_rp[1], target_y[0]])

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
        return (action+1)*self.MAX_RPM/2

    def _observationSpace(self):
        """
        OBS OF SIZE 23 (WITH QUATERNION AND RPMS)
        Observation vector -------- X  Y  Z       Q1   Q2   Q3   Q4       R  P  Y       VX VY VZ       WX WY WZ       P0 P1 P2 P3      TR, TP, TY
        """
        obs_lower_bound = np.array([-1,-1, 0,     -1,  -1,  -1,  -1,      -1,-1,-1,     -1,-1,-1,      -1,-1,-1,      0, 0, 0, 0,      -1, -1, -1])
        obs_upper_bound = np.array([ 1, 1, 1,      1,   1,   1,   1,       1, 1, 1,      1, 1, 1,       1, 1, 1,      1, 1, 1, 1,       1,  1,  1])          
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (23,).

        """
        state = self._getDroneStateVector(0)
        obs = self._clipAndNormalizeState(state)
        obs = np.hstack([obs, 
                        self.target_rpys/np.pi])
        
        return obs

    def _clipAndNormalizeState(self,
                               state
                               ):
        """Normalizes a drone's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (23,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (23,)-shaped array of floats containing the normalized state of a single drone.

        """
        MAX_LIN_VEL_XY = 3
        MAX_LIN_VEL_Z = 1

        MAX_XY = MAX_LIN_VEL_XY*self.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z*self.EPISODE_LEN_SEC

        MAX_PITCH_ROLL = np.pi # Full range

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        if self.GUI:
            self._clipAndNormalizeStateWarning(state,
                                               clipped_pos_xy,
                                               clipped_pos_z,
                                               clipped_rp,
                                               clipped_vel_xy,
                                               clipped_vel_z
                                               )

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = state[13:16]/np.linalg.norm(state[13:16]) if np.linalg.norm(state[13:16]) != 0 else state[13:16]
        normalized_rpm = state[16:20]/self.MAX_RPM

        norm_and_clipped = np.hstack([normalized_pos_xy,
                                      normalized_pos_z,
                                      state[3:7],
                                      normalized_rp,
                                      normalized_y,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel,
                                      normalized_rpm
                                      ]).reshape(20,)

        return norm_and_clipped

    def _clipAndNormalizeStateWarning(self,
                                      state,
                                      clipped_pos_xy,
                                      clipped_pos_z,
                                      clipped_rp,
                                      clipped_vel_xy,
                                      clipped_vel_z,
                                      ):
        """Debugging printouts associated to `_clipAndNormalizeState`.

        Print a warning if values in a state vector is out of the clipping range.
        
        """
        if not(clipped_pos_xy == np.array(state[0:2])).all():
            print("[WARNING] it", self.step_counter, "in AttitudeAviary2_3._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0], state[1]))
        if not(clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter, "in AttitudeAviary2_3._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not(clipped_rp == np.array(state[7:9])).all():
            print("[WARNING] it", self.step_counter, "in AttitudeAviary2_3._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[7], state[8]))
        if not(clipped_vel_xy == np.array(state[10:12])).all():
            print("[WARNING] it", self.step_counter, "in AttitudeAviary2_3._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10], state[11]))
        if not(clipped_vel_z == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter, "in AttitudeAviary2_3._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))


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