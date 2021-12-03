"""
REINFORCE on the cartpole problem.
https://www.janisklaise.com/post/rl-policy-gradients/
"""

import numpy as np 
import gym
from gym.wrappers.monitor import Monitor, load_results
from numpy.random.mtrand import gamma
import matplotlib.pyplot as plt

class LogisticPolicy:
    
    def __init__(self, theta, alpha, gamma):
        self.theta = theta
        self.alpha = alpha
        self.gamma = gamma 

    def logistic(self, y):
        return 1/(1 + np.exp(-y))

    def probs(self, x):
        y = x @ self.theta # matrix multiplication
        prob0 = self.logistic(y)
        return np.array([prob0, 1-prob0])

    def act(self, x):
        probs = self.probs(x)
        action = np.random.choice([0, 1], p=probs)
        return action, probs[action]

    def grad_log_p(self, x):
        y = x @ self.theta # matrix multiplication
        grad_log_p0 = x - x*self.logistic(y)
        grad_log_p1 = - x*self.logistic(y)
        return grad_log_p0, grad_log_p1

    def grad_log_p_dot_rewards(self, grad_log_p, actions, discounted_rewards):
        return grad_log_p.T @ discounted_rewards # matrix multiplication

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros(len(rewards))
        cumulative_rewards = 0
        for i in reversed(range(0, len(rewards))):
            cumulative_rewards = cumulative_rewards * self.gamma + rewards[i]
            discounted_rewards[i] = cumulative_rewards

        return discounted_rewards

    def update(self, rewards, obs, actions):
        grad_log_p = np.array([self.grad_log_p(ob)[action] for ob, action in zip(obs, actions)])

        assert grad_log_p.shape == (len(obs), 4)

        discounted_rewards = self.discount_rewards(rewards)

        dot = self.grad_log_p_dot_rewards(grad_log_p, actions, discounted_rewards)

        self.theta += self.alpha*dot    


def run_episode(env, policy, render=False):

    observation = env.reset()
    totalreward = 0

    observations = []
    actions = []
    rewards = []
    probs = []

    done = False

    while not done:
        if render:
            env.render()

        observations.append(observation)

        action, prob = policy.act(observation)
        observation, reward, done, info = env.step(action)

        totalreward += reward
        rewards.append(reward)
        actions.append(action)
        probs.append(prob)

    return totalreward, np.array(rewards), np.array(observations), np.array(actions), np.array(probs)

def train(theta, alpha, gamma, Policy, MAX_EPISODES=1000, seed=None, evaluate=False):

    # initialize environment and policy
    env = gym.make('CartPole-v0')
    if seed is not None:
        env.seed(seed)
    episode_rewards = []
    policy = Policy(theta, alpha, gamma)

    # train until MAX_EPISODES
    for i in range(MAX_EPISODES):

        # run a single episode
        total_reward, rewards, observations, actions, probs = run_episode(env, policy)

        # keep track of episode rewards
        episode_rewards.append(total_reward)

        # update policy
        policy.update(rewards, observations, actions)
        print("EP: " + str(i) + " Score: " + str(total_reward) + " ",end="\r", flush=False)

    # evaluation call after training is finished - evaluate last trained policy on 100 episodes
    if evaluate:
        env = Monitor(env, 'pg_cartpole/', video_callable=False, force=True)
        for _ in range(100):
            run_episode(env, policy, render=False)
        env.env.close()

    return episode_rewards, policy


if __name__ == "__main__":
    GLOBAL_SEED = 0
    np.random.seed(GLOBAL_SEED)

    episode_rewards, policy = train(theta=np.random.rand(4),
                                    alpha=0.002,
                                    gamma=0.99,
                                    Policy=LogisticPolicy,
                                    MAX_EPISODES=2000,
                                    seed=GLOBAL_SEED,
                                    evaluate=False)

    plt.plot(episode_rewards)
    plt.show()
