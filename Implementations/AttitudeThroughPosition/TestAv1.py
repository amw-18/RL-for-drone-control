import numpy as np
from scipy.spatial.transform.rotation import Rotation as R
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary
from gym import spaces
from ModDSLPID import ModDSLPIDpos
import math

class TestAv1PID(BaseAviary):
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
        self.PWM2RPM_scale = 0.2685
        self.PWM2RPM_const = 65535
        self.EPISODE_LEN_SEC = ep_len

        assert(self.DRONE_MODEL==DroneModel.CF2X)
        self.PID_pos_ctrl = ModDSLPIDpos(DroneModel.CF2X)
        self.target_z_thrust = self.GRAVITY
        self.computed_target_rpy = np.array([0., 0., 0.])
        self.last_computed_target_rpy = None
        
        self._wp_creator()

    def _wp_creator(self):
        # self.target_rpy = np.array([*(np.random.rand(2)*2-1), 0])*np.pi/6   # -pi/6 to +pi/6
        self.target_rpy = np.array([np.pi/18, np.pi/18, 0])
        target_facing = R.from_euler('xyz', self.target_rpy).apply(np.array([0., 0., 1.]))

        self.NUM_WP = self.EPISODE_LEN_SEC*self.SIM_FREQ
        self.TARGET_VEL = np.sign(target_facing)*np.linspace([0, 0, 0], np.abs(target_facing)*self.REQD_SPEED, self.NUM_WP)
        self.TARGET_POS = np.zeros((self.NUM_WP,3))
        for i in range(1, self.NUM_WP):
            self.TARGET_POS[i] = self.TARGET_POS[i-1] + self.TARGET_VEL[i-1]*(1/self.SIM_FREQ)
        
        self.wp_counter = 0

    def reset(self):
        self.PID_pos_ctrl = ModDSLPIDpos(DroneModel.CF2X)
        self.target_z_thrust = self.GRAVITY
        self.computed_target_rpy = np.array([0., 0., 0.])
        self.last_computed_target_rpy = None

        self._wp_creator()
        return super().reset()

    def _actionSpace(self):
        """
        RPM action: [P1, P2, P3, P4] 
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
        # of length 24
        #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ       P0            P1            P2            P3            Thrust           TargetR    TargetP    TargetY
        obs_lower_bound = np.array([-np.inf, -np.inf, 0.,     -1., -1., -1., -1., -np.pi, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0.,           0.,           0.,           0.,           0,               -np.pi,    -np.pi,    -np.pi])
        obs_upper_bound = np.array([np.inf,  np.inf,  np.inf, 1.,  1.,  1.,  1.,  np.pi,  np.pi,  np.pi,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_THRUST, np.pi,      np.pi,     np.pi])
        return spaces.Box(low  = obs_lower_bound,
                          high = obs_upper_bound,
                          dtype= np.float32)

    def step(self, action):
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

        return super().step(action)


    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (24,).

        """
        obs = np.hstack([self._getDroneStateVector(0),
                         self.target_z_thrust,
                         self.computed_target_rpy]).reshape(24,)
        
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
        # Add the error due to difference between required thrust and actual thrust
        z_thrust = np.sum(self.KF*(self.last_clipped_action**2))
        target_z_thrust = 4*self.KF*(self.target_z_thrust*self.PWM2RPM_scale+self.PWM2RPM_const)**2
        norm_thrust_err = np.abs(target_z_thrust-z_thrust)/(4*self.MAX_THRUST)
        return -np.sum(rpy_err) - norm_thrust_err

    def _computeInfo(self):
        return {}

class TestAv1RL(TestAv1PID):
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
                 ep_len,
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
                 dynamics_attributes
                 )
        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MAX_THRUST_PWM = (math.sqrt(self.MAX_THRUST / (4*self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE

    def _observationSpace(self):
        # of length 10
        #### Observation vector ### R       P       Y        WX       WY       WZ        Thrust           TargetR    TargetP    TargetY
        obs_lower_bound = np.array([-1, -1, -1,  -1, -1, -1,  0, -1,    -1,    -1])
        obs_upper_bound = np.array([1,  1,  1,   1,  1,  1,   1, 1,      1,     1])
        return spaces.Box(low  = obs_lower_bound,
                          high = obs_upper_bound,
                          dtype= np.float32)

    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (10,).

        """
        state = self._getDroneStateVector(0)
        norm_rpy = state[7:10]/np.pi
        rpy_rates = state[13:16]
        if np.linalg.norm(rpy_rates) > 0:
            norm_rpy_rates = rpy_rates/np.linalg.norm(rpy_rates)
        else:
            norm_rpy_rates = rpy_rates
        norm_target_z_thrust = self.target_z_thrust/self.MAX_THRUST_PWM
        norm_target_rpy = self.computed_target_rpy/np.pi
        obs = np.hstack([norm_rpy,
                         norm_rpy_rates,
                         norm_target_z_thrust,
                         norm_target_rpy]).reshape(10,)
        
        return obs

    def _actionSpace(self):
        """
        RPM action: [P1, P2, P3, P4] 
        """
        act_lower_bound = np.array([0.,           0.,           0.,           0.])
        act_upper_bound = np.array([1.,           1.,           1.,           1.])
        return spaces.Box(low  = act_lower_bound,
                          high = act_upper_bound,
                          dtype= np.float32)

    def _preprocessAction(self,
                          action
                          ):
        
        action = action*self.MAX_RPM
        return action