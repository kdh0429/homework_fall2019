import numpy as np

from .base_policy import BasePolicy


import time
class MPCPolicy(BasePolicy):

    def __init__(self, sess, env, ac_dim, dyn_models, horizon, N, **kwargs):
        super().__init__(**kwargs)

        # init vars
        self.sess = sess
        self.env = env
        self.dyn_models = dyn_models
        self.horizon = horizon
        self.N = N
        self.data_statistics = None # NOTE must be updated from elsewhere

        self.ob_dim = self.env.observation_space.shape[0]

        # action space
        self.ac_space = self.env.action_space
        self.ac_dim = ac_dim
        self.low = self.ac_space.low
        self.high = self.ac_space.high

        # PDDM
        self.action_mean = self.horizon*[self.ac_space.sample()]
        self.action_variance = np.zeros((self.N, self.horizon, self.ac_dim))
        self.beta = 0.7
        self.gamma = 10

    def sample_action_sequences(self, num_sequences, horizon):
        # TODO(Q1) uniformly sample trajectories and return an array of
        # dimensions (num_sequences, horizon, self.ac_dim)
        random_action_sequences = np.zeros((num_sequences, horizon, self.ac_dim))
        for seq_idx in range(num_sequences):
            for hor_idx in range(horizon):
                random_action_sequences[seq_idx][hor_idx] = self.ac_space.sample() 
        return random_action_sequences

    def get_action(self, obs):

        if self.data_statistics is None:
            # print("WARNING: performing random actions.")
            return self.sample_action_sequences(num_sequences=1, horizon=1)[0]

        #sample random actions (Nxhorizon)
        candidate_action_sequences = self.sample_action_sequences(num_sequences=self.N, horizon=self.horizon)

        # a list you can use for storing the predicted reward for each candidate sequence
        predicted_rewards_ens = []

        for model in self.dyn_models:
            # TODO(Q2)
            # for each candidate action sequence, predict a sequence of
            # states for each dynamics model in your ensemble
            ob = np.tile(obs,[self.N,1])
            for hor_idx in range(self.horizon):
                rew, _ = self.env.get_reward(ob, candidate_action_sequences[:,hor_idx,:])
                predicted_rewards_ens.append(rew)
                ob = model.get_prediction(ob, candidate_action_sequences[:,hor_idx,:], self.data_statistics)
            # once you have a sequence of predicted states from each model in your
            # ensemble, calculate the reward for each sequence using self.env.get_reward (See files in envs to see how to call this)

        # pick the action sequence and return the 1st element of that sequence
        predicted_rewards_ens_total = np.mean(predicted_rewards_ens, axis=0)
        best_index = np.argmax(predicted_rewards_ens_total) #TODO(Q2)
        best_action_sequence = candidate_action_sequences[best_index] #TODO(Q2)
        action_to_take = best_action_sequence[0] # TODO(Q2)
        return action_to_take[None] # the None is for matching expected dimensions

        # # PDDM
        # update_iter = 10
        # self.action_mean = self.horizon*[np.zeros_like(self.ac_space.sample())]
        # self.action_variance = np.zeros((self.N, self.horizon, self.ac_dim))
        # for itr in range(update_iter):
        #     # sample 추출                 
        #     candidate_action_sequences = np.zeros((self.N, self.horizon, self.ac_dim))
        #     for seq_idx in range(self.N):
        #         for hor_idx in range(self.horizon):
        #             if hor_idx > 0:
        #                 self.action_variance[seq_idx][hor_idx] = self.beta * np.random.normal(0,1) + (1-self.beta) * self.action_variance[seq_idx][hor_idx-1]
        #             if itr == 0:
        #                 candidate_action_sequences[seq_idx][hor_idx] = self.ac_space.sample() + self.action_variance[seq_idx][hor_idx]
        #             else:
        #                 candidate_action_sequences[seq_idx][hor_idx] = self.action_mean[hor_idx] + self.action_variance[seq_idx][hor_idx]

        #     # mean, variace update
        #     for model in self.dyn_models:
        #         ob = np.tile(obs,[self.N,1])
        #         for hor_idx in range(self.horizon):
        #             rew, _ = self.env.get_reward(ob, candidate_action_sequences[:,hor_idx,:])
        #             ob = model.get_prediction(ob, candidate_action_sequences[:,hor_idx,:], self.data_statistics)
        #             self.action_mean[hor_idx] += np.mean(np.exp(self.gamma*np.array(rew).reshape((self.N,1))) * candidate_action_sequences[:,hor_idx,:] / (sum(np.exp(self.gamma*rew)))*len(self.dyn_models), axis=0)

        # action_to_take = self.action_mean[0] # TODO(Q2)
        # return action_to_take[None] # the None is for matching expected dimensions
