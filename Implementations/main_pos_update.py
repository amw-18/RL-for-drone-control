from numpy.lib.function_base import trim_zeros
import pygame
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import *
from gym_pybullet_drones.control.DSLPIDControl import *
from gym_pybullet_drones.utils.utils import sync
from pygame.locals import *
import time


def get_action(pressed_keys):
    add_pos = np.array([0., 0., 0.])
    if pressed_keys[K_UP]:
        add_pos[1] += 0.1
    if pressed_keys[K_DOWN]:
        add_pos[1] -= 0.1
    if pressed_keys[K_RIGHT]:
        add_pos[0] += 0.1
    if pressed_keys[K_LEFT]:
        add_pos[0] -= 0.1

    return add_pos


if __name__ == "__main__":
    simulation_freq_hz = 240
    control_freq_hz = 240
    aggregate = True
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
    target_pos = INIT_XYZS
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
                add_pos = get_action(pressed)
                target_pos[0] = obs[str(0)]["state"][0:3] + add_pos

            action[str(0)], _, _ = ctrl[0].computeControlFromState(control_timestep=CTRL_EVERY_N_STEPS*env.TIMESTEP,
                                                                    state = obs[str(0)]["state"],
                                                                    target_pos=target_pos[0]                                                      
                                                                    )

        if STEP%(env.SIM_FREQ/1) == 0:
            env.render()
        
        sync(STEP, START, env.TIMESTEP)
        STEP += 1
    
    env.close()