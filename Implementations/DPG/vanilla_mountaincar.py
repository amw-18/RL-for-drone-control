import gym
import numpy as np
import matplotlib.pyplot as plt 

class DeterministicPolicyMCar:
    def __init__(self):
        self.discount = 1
        self.rate_critic = 0.01
        self.rate_actor = 0.01
        self.actor_params = np.random.normal(0, 1, 3)
        self.critic_params = np.random.normal(0, 1, 4)

    def action(self, state):
        """
        power-coeff: [-1, +1]
        """
        p = np.dot(self.actor_params[:2], state) + self.actor_params[2]
        return np.tanh(p)

    def q_value(self, state, action):
        """
        Linear w.r.t. state and action
        """
        return np.dot(self.critic_params[:2], state) + self.critic_params[2]*action + self.critic_params[3]


    def update(self, s0, a0, r, s1, a1):
        # SARSA critic update
        diff = r + self.discount*self.q_value(s1, a1) - self.q_value(s0, a0)
        grad_critic = np.array([*s0, a0, 1])
        self.critic_params += self.rate_critic*diff*grad_critic
        
        # Vanilla actor update
        grad_actor = (1-a0**2)*np.array([*s0, 1])
        self.actor_params += self.rate_actor*grad_actor*self.critic_params[2]


if __name__ == "__main__":
    max_episodes = 1000
    env = gym.make("MountainCarContinuous-v0")
    policy = DeterministicPolicyMCar()
    ep = []
    Gs = []
    for i_episode in range(max_episodes):
        G = 0
        s0 = env.reset()
        a0 = policy.action(s0)
        ep_len = 0
        while True:
            s1, r, done, info = env.step([a0])
            G += r
            ep_len += 1
            if done:
                # print(f"Episode {i_episode} finished after {ep_len} steps.")
                break

            a1 = policy.action(s1)

            # update
            policy.update(s0, a0, r, s1, a1)

            s0 = s1
            a0 = a1

            
        ep.append(ep_len)
        Gs.append(G)
        
    plt.plot(ep)
    plt.plot(Gs)
    plt.show()


