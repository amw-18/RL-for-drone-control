from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import *


class AltHoldAviary(HoverAviary):
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
        r = abs(self.INIT_XYZS[0][2] - state[2])  # altitude difference
        return -r

    def _computeDone(self):
        return super()._computeDone()