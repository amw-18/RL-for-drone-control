import gym
import numpy as np
import matplotlib.pyplot as plt 
from copy import deepcopy

class DeterministicPolicy:
    def __init__(self, params):
        self.theta = params

    def __call__(self, state):
        return np.tanh(np.dot(self.theta, state))


class SValueFunction:
    def __init__(self, params):
        self.v = params

    def __call__(self, state):
        return np.dot(self.v, state)


class QValueFunction:
    def __init__(self, params):
        self.w = params

    def __call__(self, state, action):
        grad = state  # gradient of policy w.r.t. policy parameters

        phi = (action - policy(state))*grad

        advantage = np.dot(phi, self.w)

        return advantage + V(state)

def update_params(state, action, reward, new_state):
    new_action = policy(state)
    td_error = reward + gamma*Q(new_state, new_action) - Q(state, action)
    grad = state
    # policy.theta += alpha_theta*(grad*np.dot(grad, Q.w))
    policy.theta += alpha_theta*(Q.w)
    phi = (action - policy(state))*grad
    Q.w += alpha_w*(td_error*phi)
    V.v += alpha_v*(td_error*grad)



max_steps = 10000
max_episodes = 10
env = gym.make("MountainCarContinuous-v0").env
env2 = deepcopy(env) # test-env

# Hyperparams setting
gamma = 0.99 # discount rate
alpha_theta = 0.005 # policy learning rate
alpha_w = 0.03 # Q learning rate
alpha_v = 0.03 # V learning rate

#environment parameters
obs_space = env.observation_space.shape[0]
action_space = env.action_space.shape[0]

# Initializing params
theta = np.random.normal(0, 1, obs_space)
w = np.random.normal(0, 1, obs_space)
v = np.random.normal(0, 1, obs_space)

# Defining Policy and Value functions
policy = DeterministicPolicy(theta)
Q = QValueFunction(w)
V = SValueFunction(v)

# Totals
totals = []
for i_episode in range(max_episodes):
    state = env.reset()
    ep_len = 0
    total = 0
    for step in range(max_steps):
        action = env.action_space.sample()[0]
        new_state, reward, done, _ = env.step([action])
        total += reward
        if done:
            break
        ep_len += 1
        # update
        update_params(state, action, reward, new_state)
        state = new_state

    print(f"Episode {i_episode} finished after {ep_len} steps and total reward {total}")
    totals.append(total)
print(max(totals))

test_episodes = 10
ep = []
totals = []
for i_episode in range(test_episodes):
    state = env2.reset()
    ep_len = 0
    total = 0
    for step in range(max_steps):
        action = policy(state)
        new_state, reward, done, _ = env2.step([action])
        total += reward
        if done:
            break
        ep_len += 1
        state = new_state
    print(f"Episode {i_episode} finished after {ep_len} steps and total reward {total}")
    ep.append(ep_len)
    totals.append(total)

print(max(totals))
    

