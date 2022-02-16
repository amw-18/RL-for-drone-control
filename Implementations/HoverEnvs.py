from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import *

class HoverR1D1(HoverAviary):
    """
    Minimize deviation from target hover position
    """
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 freq: int=240,
                 aggregate_phy_steps: int=1,
                 gui=False,
                 record=False, 
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM
                 ):
        super().__init__(drone_model=drone_model,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )

    def _computeReward(self):
        return super()._computeReward()

    def _computeDone(self):
        return super()._computeDone()


class HoverR2D2(HoverAviary):
    """
    
    """
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 freq: int=240,
                 aggregate_phy_steps: int=1,
                 gui=False,
                 record=False, 
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM
                 ):
        super().__init__(drone_model=drone_model,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        r = np.linalg.norm(self.INIT_XYZS[0]-state[0:3])
        if 0 < r < 0.15:
            return min(1/r, 100)
        else:
            return -r

    def _computeDone(self):
        state = self._getDroneStateVector(0)
        r = np.linalg.norm(self.INIT_XYZS[0]-state[0:3])
        if (r > 0.5) or (self.step_counter/self.SIM_FREQ > self.EPISODE_LEN_SEC):
            return True
        else:
            return False


class HoverR3D1(HoverAviary):
    """
    Maximize the time of each episode along with incentivizing to minimize velocities and drift
    """
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 freq: int=240,
                 aggregate_phy_steps: int=1,
                 gui=False,
                 record=False, 
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM
                 ):
        super().__init__(drone_model=drone_model,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        return 0 - np.linalg.norm(state[10:13]) - np.linalg.norm(state[13:16]) - np.linalg.norm(self.INIT_XYZS[0]-state[0:3])

    def _computeDone(self):
        return super()._computeDone()


class HoverR3D2(HoverR3D1):
    """
    Episode ends when angular velocities break a threshold
    """
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 freq: int=240,
                 aggregate_phy_steps: int=1,
                 gui=False,
                 record=False, 
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM
                 ):
        super().__init__(drone_model=drone_model,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )

    def _computeReward(self):
        return super()._computeReward()

    def _computeDone(self):
        state = self._getDroneStateVector(0)
        r = np.linalg.norm(self.INIT_XYZS[0]-state[0:3])
        if (r > 0.5) or (self.step_counter/self.SIM_FREQ > self.EPISODE_LEN_SEC):
            return True
        else:
            return False

class HoverR4D2(HoverR2D2):
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 freq: int=240,
                 aggregate_phy_steps: int=1,
                 gui=False,
                 record=False, 
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM
                 ):
        super().__init__(drone_model=drone_model,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        r = np.linalg.norm(self.INIT_XYZS[0]-state[0:3])
        if 0 < r < 0.15:
            return min(0.1/r, 10) - np.linalg.norm(state[10:13]) - np.linalg.norm(state[13:16])
        else:
            return - r 

    def _computeDone(self):
        return super()._computeDone()