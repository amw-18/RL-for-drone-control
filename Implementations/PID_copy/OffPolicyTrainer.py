from ray.rllib.execution.replay_buffer import ReplayBuffer
from ray.rllib.evaluation.rollout_worker import RolloutWorker

class OffPolicyTrainer:
    def __init__(self,
                 env_creator, 
                 buffer_size,
                 batch_size,
                 behavior_policy, 
                 rl_policy):

        # Replay Buffer
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size

        # Creating relevant environments
        self.worker = RolloutWorker(env_creator=env_creator, 
                                    policy_spec=behavior_policy, 
                                    batch_mode="complete_episodes")
        self.eval_worker = RolloutWorker(env_creator=env_creator, 
                                    policy_spec=rl_policy, 
                                    batch_mode="complete_episodes")

    def learn(self):
        # Run a new episode and collect samples
        new_batch = self.worker.sample()
        self.buffer.add(new_batch, weight=None)

        # Extract samples from Replay Buffer
        if len(self.buffer) > self.batch_size:
            learn_batch = self.buffer.sample(self.batch_size)

            self.eval_worker.learn_on_batch(learn_batch)

    def evaluate(self):
        eval_episode = self.eval_worker.sample()
        return eval_episode

    

    

        
