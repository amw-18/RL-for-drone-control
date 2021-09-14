from numpy.lib.function_base import trim_zeros
import pygame
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import *
from gym_pybullet_drones.control.DSLPIDControl import *
from gym_pybullet_drones.utils.utils import sync
from pygame.locals import *
import pybullet as p
import time


def get_action(pressed_keys):
    add_angle = np.array([0., 0., 0.])
    if pressed_keys[K_UP]:
        add_angle[1] += 0.0087
    if pressed_keys[K_DOWN]:
        add_angle[1] -= 0.0087
    if pressed_keys[K_RIGHT]:
        add_angle[0] += 0.0087
    if pressed_keys[K_LEFT]:
        add_angle[0] -= 0.0087

    return add_angle


if __name__ == "__main__":
    simulation_freq_hz = 240
    control_freq_hz = 48
    aggregate = False
    # Initializing the simulation
    INIT_XYZS = np.array([[0., 0., 1.]])
    INIT_RPYS = np.array([[0., 0., 0.]])
    AGGR_PHY_STEPS = int(simulation_freq_hz/control_freq_hz) if aggregate else 1

    env = CtrlAviary(drone_model=DroneModel.CF2X,
                    num_drones=1,
                    initial_xyzs=INIT_XYZS,
                    initial_rpys=INIT_RPYS,
                    neighbourhood_radius=10,
                    freq=simulation_freq_hz,
                    aggregate_phy_steps=AGGR_PHY_STEPS,
                    gui=True,
                    record=False,
                    obstacles=True,
                    user_debug_gui=False
                    )

    PYB_CLIENT = env.getPyBulletClient()
    
    ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X)]

    pygame.init()
    screen = pygame.display.set_mode((300, 100))
    pygame.display.set_caption('Drone Control Test')
    pygame.mouse.set_visible(0)

    action = {str(0): np.array([0,0,0,0])}
    START = time.time()
    STEP = 0
    running = True
    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ/control_freq_hz))
    while running:
        obs, reward, done, info = env.step(action)

        if STEP%CTRL_EVERY_N_STEPS == 0:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False

            # If keyboard is pressed, modify action
            pressed = pygame.key.get_pressed()
            if pressed:
                add_angle = get_action(pressed)
                cur_rpy = np.array(p.getEulerFromQuaternion(obs[str(0)]["state"][3:7]))
                #TODO: add proper code to determine target_rpy
                # if cur_rpy + add_angle > 1:
                #     target_rpy
                # target_rpy[0] =  + add_angle

            action[str(0)], _, _ = ctrl[0].computeControlFromState(control_timestep=CTRL_EVERY_N_STEPS*env.TIMESTEP,
                                                                    state=obs[str(0)]["state"],
                                                                    target_pos=obs[str(0)]["state"][0:3],
                                                                    target_rpy=                                                     
                                                                    )

        if STEP%(env.SIM_FREQ/1) == 0:
            env.render()
        
        sync(STEP, START, env.TIMESTEP)
        STEP += 1
    
    env.close()