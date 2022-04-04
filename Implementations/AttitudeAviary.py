import numpy as np
from gym import spaces

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType, BaseSingleAgentAviary
from gym import spaces


class AttitudeAviary1(BaseSingleAgentAviary):
    """
    Environment for learning attitude control. Episodes of length 1 sec. 
    Episodes end when a threshold angular velocity is reached.
    Higher reward for being closer to specified target angular velocities.
    Action : (4,) -> (p0, p1, p2, p3) [motor rpms]
    Observation : (12,) -> (x, y, z, r, p, y, vx, vy, vz, wx, wy, wz)
    """
    def __init__(self, 
                drone_model: DroneModel = DroneModel.CF2X,  
                physics: Physics = Physics.PYB, 
                freq: int = 1000, 
                aggregate_phy_steps: int = 1, 
                gui=False, 
                record=False):
        super().__init__(drone_model, 
                        np.array([[0.0, 0.0, 1.0]]), # initial_xyzs
                        np.array([[0.0, 0.0, 0.0]]), # initial_rpys
                        physics, 
                        freq, 
                        aggregate_phy_steps, 
                        gui, 
                        record, 
                        ObservationType.KIN, # Kinematic
                        ActionType.RPM) # Individual RPMs to motors

        self.max_rpy_rates = 5.24 # rad/s

        self.EPISODE_LEN_SEC = 1

        self.target_rpy_rates = self._sample_rpy_rates()

    def reset(self):
        self.target_rpy_rates = self._sample_rpy_rates()
        return super().reset()

    def _sample_rpy_rates(self):
        rpy_rates = spaces.Box(-self.max_rpy_rates, self.max_rpy_rates, (3,)).sample()
        return np.array([rpy_rates])

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        current_rpy_rates = state[13:16]
        return -1*np.clip(np.sum(np.abs(self.target_rpy_rates - current_rpy_rates))/ \
            (3*self.max_rpy_rates), 0, 1)

    # def _computeDone(self):
    #     state = self._getDroneStateVector(0)
    #     current_rpy_rates = state[13:16]
    #     if (self.step_counter/self.SIM_FREQ > self.EPISODE_LEN_SEC) or \
    #     (np.max(np.abs(current_rpy_rates)) > self.max_rpy_rates):
    #         return True
    #     else:
    #         return False

    def _computeDone(self):
        return (self.step_counter/self.SIM_FREQ > self.EPISODE_LEN_SEC)

    def _computeInfo(self):
        return {}

    def _clipAndNormalizeState(self,
                               state
                               ):
        """Normalizes a drone's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.

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

        norm_and_clipped = np.hstack([normalized_pos_xy,
                                      normalized_pos_z,
                                      state[3:7],
                                      normalized_rp,
                                      normalized_y,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel,
                                      state[16:20]
                                      ]).reshape(20,)

        return norm_and_clipped
    
    ################################################################################
    
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
            print("[WARNING] it", self.step_counter, "in AttitudeAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0], state[1]))
        if not(clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter, "in AttitudeAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not(clipped_rp == np.array(state[7:9])).all():
            print("[WARNING] it", self.step_counter, "in AttitudeAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[7], state[8]))
        if not(clipped_vel_xy == np.array(state[10:12])).all():
            print("[WARNING] it", self.step_counter, "in AttitudeAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10], state[11]))
        if not(clipped_vel_z == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter, "in AttitudeAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))


class AttitudeAviary1_1(BaseSingleAgentAviary):
    def __init__(self, 
                drone_model: DroneModel = DroneModel.CF2X, 
                initial_xyzs=np.array([[0., 0., 1.]]), 
                initial_rpys=np.array([[0., 0., 0.]]), 
                physics: Physics = Physics.PYB, 
                freq: int = 1000, 
                aggregate_phy_steps: int = 1, 
                gui=False, 
                record=False, 
                obs: ObservationType = ObservationType.KIN, 
                act: ActionType = ActionType.RPM):
        super().__init__(drone_model, 
                        initial_xyzs, 
                        initial_rpys, 
                        physics, 
                        freq, 
                        aggregate_phy_steps, 
                        gui, 
                        record, 
                        obs, 
                        act)

        self.max_rpy_rates = np.array([5.24]*3)  # rad/s

        self.max_rpy = np.array([np.pi/4, np.pi/4, np.pi], dtype=np.float32)  # rad/s

        self.EPISODE_LEN_SEC = 1

        self.rpy_space = spaces.Box(-self.max_rpy, self.max_rpy, (3,))

        self.target_rpy = self._sample_rpy()

    def _sample_rpy(self):
        return self.rpy_space.sample()

    def reset(self):
        self.target_rpy = self._sample_rpy()
        return super().reset()

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        rpy = state[7:10]
        rpy_rates = state[13:16]
        
        if (np.abs(rpy_rates) > self.max_rpy_rates).any():
            rpy_rates_loss = -1000
        else:
            rpy_rates_loss = 0

        rpy_loss = -np.sum(np.abs((self.target_rpy - rpy)/self.max_rpy))/3

        return rpy_loss + rpy_rates_loss
    
    def _computeDone(self):
        state = self._getDroneStateVector(0)
        rpy_rates = state[13:16]
        if (np.abs(rpy_rates) > self.max_rpy_rates).any():
            return True
        else:
            return (self.step_counter/self.SIM_FREQ > self.EPISODE_LEN_SEC)

    def _computeInfo(self):
        return {}

    def _clipAndNormalizeState(self,
                               state
                               ):
        """Normalizes a drone's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.

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

        norm_and_clipped = np.hstack([normalized_pos_xy,
                                      normalized_pos_z,
                                      state[3:7],
                                      normalized_rp,
                                      normalized_y,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel,
                                      state[16:20]
                                      ]).reshape(20,)

        return norm_and_clipped
    
    ################################################################################
    
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
            print("[WARNING] it", self.step_counter, "in AttitudeAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0], state[1]))
        if not(clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter, "in AttitudeAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not(clipped_rp == np.array(state[7:9])).all():
            print("[WARNING] it", self.step_counter, "in AttitudeAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[7], state[8]))
        if not(clipped_vel_xy == np.array(state[10:12])).all():
            print("[WARNING] it", self.step_counter, "in AttitudeAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10], state[11]))
        if not(clipped_vel_z == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter, "in AttitudeAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))

