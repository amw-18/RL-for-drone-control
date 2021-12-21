import gym
import numpy as np
import matplotlib.pyplot as plt 

class DeterministicPolicyCartpole:
    def __init__(self):
        self.discount = 1
        self.rate_critic = 0.1
        self.critic_params = np.random.normal(0, 1, 6)

    def action(self, state):
        """
        +1 for right
        -1 for left
        """
        right = self.q_value(state, 1)
        left = self.q_value(state, 0)
        if right > left:
            return 1
        else:
            return 0

    def q_value(self, state, action):
        """
        Linear w.r.t. state and action
        """
        return np.dot(self.critic_params[:4], state) + self.critic_params[4]*action + self.critic_params[5]


    def update(self, s0, a0, r, s1, a1):
        # SARSA critic update
        diff = r + self.discount*self.q_value(s1, a1) - self.q_value(s0, a0)
        grad_critic = np.array([*s0, a0, 1])
        self.critic_params += self.rate_critic*diff*grad_critic


if __name__ == "__main__":
    max_episodes = 200
    env = gym.make("CartPole-v0")
    policy = DeterministicPolicyCartpole()
    ep = []
    for i_episode in range(max_episodes):
        s0 = env.reset()
        a0 = policy.action(s0)
        ep_len = 0
        while True:
            s1, r, done, info = env.step(a0)
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
        
        
    plt.plot(ep)
    plt.show()


