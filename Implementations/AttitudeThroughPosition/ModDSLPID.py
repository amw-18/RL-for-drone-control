import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from gym_pybullet_drones.envs.BaseAviary import DroneModel
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl


class ModDSLPID(DSLPIDControl):
    def __init__(self, drone_model: DroneModel, g: float = 9.8):
        super().__init__(drone_model, g)

    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3)
                       ):
        """Computes the PID control action (as RPMs) for a single drone.

        This methods sequentially calls `_dslPIDPositionControl()` and `_dslPIDAttitudeControl()`.
        Parameter `cur_ang_vel` is unused.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        cur_ang_vel : ndarray
            (3,1)-shaped array of floats containing the current angular velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray, optional
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray, optional
            (3,1)-shaped array of floats containing the desired velocity.
        target_rpy_rates : ndarray, optional
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.
        ndarray
            (3,1)-shaped array of floats containing the current XYZ position error.
        float
            The current computed rpy

        """
        self.control_counter += 1
        thrust, computed_target_rpy, pos_e = self._dslPIDPositionControl(control_timestep,
                                                                         cur_pos,
                                                                         cur_quat,
                                                                         cur_vel,
                                                                         target_pos,
                                                                         target_rpy,
                                                                         target_vel
                                                                         )
        rpm = self._dslPIDAttitudeControl(control_timestep,
                                          thrust,
                                          cur_quat,
                                          computed_target_rpy,
                                          target_rpy_rates
                                          )

        return rpm, pos_e, computed_target_rpy

class ModDSLPIDpos(DSLPIDControl):
    def __init__(self, drone_model: DroneModel, g: float = 9.8):
        super().__init__(drone_model, g)

    def computeControl(self,
                       control_timestep,
                       state,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3)
                       ):
        cur_pos = state[:3]
        cur_quat = state[3:7]
        cur_vel = state[10:13]

        self.control_counter += 1
        thrust, computed_target_rpy, pos_e = self._dslPIDPositionControl(control_timestep,
                                                                         cur_pos,
                                                                         cur_quat,
                                                                         cur_vel,
                                                                         target_pos,
                                                                         target_rpy,
                                                                         target_vel
                                                                         )

        return thrust, computed_target_rpy

class ModDSLPIDatt(DSLPIDControl):
    def __init__(self, drone_model: DroneModel, g: float = 9.8):
        super().__init__(drone_model, g)

    def computeControl(self,
                       control_timestep,
                       state,
                       thrust, 
                       computed_target_rpy,
                       target_rpy_rates=np.zeros(3)
                       ):
        cur_quat = state[3:7]
        return self._dslPIDAttitudeControl(control_timestep,
                                          thrust,
                                          cur_quat,
                                          computed_target_rpy,
                                          target_rpy_rates
                                          )

class ModDSLPIDatt_torq(ModDSLPIDatt):
    def __init__(self, drone_model: DroneModel, g: float = 9.8):
        super().__init__(drone_model, g)

    def computeControl(self,
                       control_timestep,
                       state, 
                       computed_target_rpy,
                       target_rpy_rates=np.zeros(3)
                       ):
        cur_quat = state[3:7]
        return self._dslPIDAttitudeControl(control_timestep,
                                          cur_quat,
                                          computed_target_rpy,
                                          target_rpy_rates
                                          )
    
    def _dslPIDAttitudeControl(self,
                               control_timestep,
                               cur_quat,
                               target_euler,
                               target_rpy_rates
                               ):
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
        target_quat = (Rotation.from_euler('XYZ', target_euler, degrees=False)).as_quat()
        w,x,y,z = target_quat
        target_rotation = (Rotation.from_quat([w, x, y, z])).as_matrix()
        rot_matrix_e = np.dot((target_rotation.transpose()),cur_rotation) - np.dot(cur_rotation.transpose(),target_rotation)
        rot_e = np.array([rot_matrix_e[2, 1], rot_matrix_e[0, 2], rot_matrix_e[1, 0]]) 
        rpy_rates_e = target_rpy_rates - (cur_rpy - self.last_rpy)/control_timestep
        self.last_rpy = cur_rpy
        self.integral_rpy_e = self.integral_rpy_e - rot_e*control_timestep
        self.integral_rpy_e = np.clip(self.integral_rpy_e, -1500., 1500.)
        self.integral_rpy_e[0:2] = np.clip(self.integral_rpy_e[0:2], -1., 1.)
        #### PID target torques ####################################
        target_torques = - np.multiply(self.P_COEFF_TOR, rot_e) \
                         + np.multiply(self.D_COEFF_TOR, rpy_rates_e) \
                         + np.multiply(self.I_COEFF_TOR, self.integral_rpy_e)
                    
        return target_torques