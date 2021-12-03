import numpy as np
import matplotlib.pyplot as plt 
from scipy.special import softmax

from gridworld import GridWorld


def generate_episode(env, p):
    env.reset()

    rewards = []
    actions = []

    while True:
        action = np.random.choice(a=[0, 1], p=p)
        state, reward = env.step(action)
        rewards.append(reward)
        actions.append(action)
        if state == 1:
            break

    return actions, rewards

def disc_return(rewards, discount):
    G = 0
    for i, reward in enumerate(rewards):
        G += (discount**i)*reward

    return G

n_test = 50
n_episodes = 1000

discount = 1
alpha = 1/(2**11)

episodic_returns = np.zeros((n_episodes,))
final_p = np.zeros((2,))
for n in range(n_test):
    env = GridWorld()
    theta = [0, 0]
    for i in range(n_episodes):
        p = softmax(theta)
        # print(p)
        actions, rewards = generate_episode(env, p)
        episode_size = len(rewards)

        for j in range(episode_size):
            G = disc_return(rewards[j:], discount)
            if j == 0:
                episodic_returns[i] += G   

            if actions[j] == 0:
                theta[0] += alpha*(discount**j)*G*(1 - p[0])
            else:
                theta[1] += alpha*(discount**j)*G*(1 - p[1])

    final_p += p

print(final_p/n_test)
plt.plot(episodic_returns/n_test)
plt.xlabel('episode')
plt.ylabel('return at start state')
plt.title("REINFORCE")
plt.grid()
plt.show()