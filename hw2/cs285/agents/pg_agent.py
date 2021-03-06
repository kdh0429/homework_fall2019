import numpy as np 

from .base_agent import BaseAgent
from cs285.policies.MLP_policy import MLPPolicyPG
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *

class PGAgent(BaseAgent):
    def __init__(self, sess, env, agent_params):
        super(PGAgent, self).__init__()

        # init vars
        self.env = env 
        self.sess = sess
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']
        self.nn_baseline = self.agent_params['nn_baseline'] 
        self.reward_to_go = self.agent_params['reward_to_go']
        self.gae = self.agent_params['gae']
        self.gae_gamma = self.agent_params['gae_gamma']
        self.gae_lambda = self.agent_params['gae_lambda']

        # actor/policy
        # NOTICE that we are using MLPPolicyPG (hw2), instead of MLPPolicySL (hw1)
            # which indicates similar network structure (layout/inputs/outputs), 
            # but differences in training procedure 
            # between supervised learning and policy gradients
        self.actor = MLPPolicyPG(sess, 
                                 self.agent_params['ac_dim'],
                                 self.agent_params['ob_dim'],
                                 self.agent_params['n_layers'],
                                 self.agent_params['size'],
                                 discrete=self.agent_params['discrete'],
                                 learning_rate=self.agent_params['learning_rate'],
                                 nn_baseline=self.agent_params['nn_baseline']
                                 ) 

        # replay buffer
        self.replay_buffer = ReplayBuffer(1000000)

    def train(self, obs, acs, rews_list, next_obs, terminals):

        """
            Training a PG agent refers to updating its actor using the given observations/actions
            and the calculated qvals/advantages that come from the seen rewards.

            ---------------------------------------------------------------------------------- 
            
            Recall that the expression for the policy gradient PG is
            
                PG = E_{tau} [sum_{t=0}^{T-1} grad log pi(a_t|s_t) * (Q_t - b_t )]
            
                where 
                tau=(s_0, a_0, s_1, a_1, s_2, a_2, ...) is a trajectory,
                Q_t is the Q-value at time t, Q^{pi}(s_t, a_t),
                b_t is a baseline which may depend on s_t,
                and (Q_t - b_t ) is the advantage.

            Thus, the PG update performed by the actor needs (s_t, a_t, q_t, adv_t),
                and that is exactly what this function provides.

            ----------------------------------------------------------------------------------
        """

        # step 1: calculate q values of each (s_t, a_t) point, 
        # using rewards from that full rollout of length T: (r_0, ..., r_t, ..., r_{T-1})
        q_values = self.calculate_q_vals(rews_list)

        # step 2: calculate advantages that correspond to each (s_t, a_t) point

        if not self.gae :
            advantage_values = self.estimate_advantage(obs, q_values)
        else:
            assert self.nn_baseline, 'GAE should be used with value function estimator. Use --nn_baseline option together.'
            advantage_values = self.estimate_generalized_advantate(obs, rews_list, next_obs, terminals, q_values)

        # step 3:
        # TODO: pass the calculated values above into the actor/policy's update, 
        # which will perform the actual PG update step
        loss = self.actor.update(obs, acs, qvals=q_values, adv_n=advantage_values)
        return loss

    def calculate_q_vals(self, rews_list):

        """
            Monte Carlo estimation of the Q function.

            arguments:
                rews_list: length: number of sampled rollouts
                    Each element corresponds to a particular rollout,
                    and contains an array of the rewards for every step of that particular rollout

            returns:
                q_values: shape: (sum/total number of steps across the rollouts)
                    Each entry corresponds to the estimated q(s_t,a_t) value 
                    of the corresponding obs/ac point at time t.
 
        """

        # Case 1: trajectory-based PG 
        if not self.reward_to_go:
            # TODO: Estimate the Q value Q^{pi}(s_t, a_t) using rewards from that entire trajectory
            # HINT1: value of each point (t) = total discounted reward summed over the entire trajectory (from 0 to T-1)
                # In other words, q(s_t, a_t) = sum_{t'=0}^{T-1} gamma^t' r_{t'}
            # Hint3: see the helper functions at the bottom of this file
            q_values = np.concatenate([self._discounted_return(r) for r in rews_list]) # N개(Not fixed) trajectory들의 q value

        # Case 2: reward-to-go PG 
        else:
            # TODO: Estimate the Q value Q^{pi}(s_t, a_t) as the reward-to-go
            # HINT1: value of each point (t) = total discounted reward summed over the remainder of that trajectory (from t to T-1)
                # In other words, q(s_t, a_t) = sum_{t'=t}^{T-1} gamma^(t'-t) * r_{t'}
            # Hint3: see the helper functions at the bottom of this file
            q_values = np.concatenate([self._discounted_cumsum(r) for r in rews_list]) # N개(Not fixed) trajectory들의 reward to go

        return q_values

    def estimate_advantage(self, obs, q_values):

        """
            Computes advantages by (possibly) subtracting a baseline from the estimated Q values
        """

        # TODO: Estimate the advantage when nn_baseline is True
        # HINT1: pass obs into the neural network that you're using to learn the baseline
            # extra hint if you're stuck: see your actor's run_baseline_prediction
        # HINT2: advantage should be [Q-b]
        # 여기서 NN은 baseline(value function)의 분산의 정도(?)를 학습(pg_agent에서 self.actor.update(obs, acs, qvals=q_values, adv_n=advantage_values)에서 qvals를 인자로 주지만 MLP_policy에서 보면 target은 이 qvals를 normalize함). 이 baseline은 학습 시 single trajectory가 아니라 N개(Not fixed) trajectory들의 평균을 사용. 1개 trajectory를 사용하여 baseline을 구하는 vanilla PG와 여태 모든 trajectory들을 학습하여 baseline을 구하는 actor-acritic의 중간처럼 보임.
        if self.nn_baseline:
            b_n_unnormalized = self.actor.run_baseline_prediction(obs) 
            b_n = b_n_unnormalized * np.std(q_values) + np.mean(q_values)
            adv_n = q_values - b_n

        # Else, just set the advantage to [Q]
        else:
            adv_n = q_values.copy()

        # Normalize the resulting advantages
        if self.standardize_advantages:
            adv_n = (adv_n - np.mean(adv_n)) / (np.std(adv_n) + 1e-8)

        return adv_n

    def estimate_generalized_advantate(self, obs, rewards, next_obs, terminals, q_values):
        rew_concat = np.concatenate([r for r in rewards])
        b_n_unnormalized = self.actor.run_baseline_prediction(obs) 
        V = b_n_unnormalized * np.std(q_values) + np.mean(q_values)

        gae = np.zeros_like(q_values)
        end_idx = 0
        
        for idx in reversed(range(len(rew_concat))):
            if terminals[idx]:
                delta = rew_concat[idx] - V[idx]
                gae[idx] = delta
                end_idx = idx
            else:
                delta = rew_concat[idx] + self.gae_gamma*V[idx+1] - V[idx]
                gae[idx] = delta + self.gae_gamma*self.gae_lambda*gae[idx+1]
        return gae



    #####################################################
    #####################################################

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size, concat_rew=False)

    #####################################################
    ################## HELPER FUNCTIONS #################
    #####################################################

    # TODO: implement this function
    def _discounted_return(self, rewards):
        """
            Helper function

            Input: a list of rewards {r_0, r_1, ..., r_t', ... r_{T-1}} from a single rollout of length T

            Output: list where each index t contains sum_{t'=0}^{T-1} gamma^t' r_{t'}
                note that all entries of this output are equivalent
                because each index t is a sum from 0 to T-1 (and doesnt involve t)
        """

        # 1) create a list of indices (t'): from 0 to T-1
        indices = np.arange(rewards.size)

        # 2) create a list where the entry at each index (t') is gamma^(t')
        discounts = np.power(self.gamma, indices)

        # 3) create a list where the entry at each index (t') is gamma^(t') * r_{t'}
        discounted_rewards = discounts*rewards

        # 4) calculate a scalar: sum_{t'=0}^{T-1} gamma^(t') * r_{t'}
        sum_of_discounted_rewards = sum(discounted_rewards)

        # 5) create a list of length T-1, where each entry t contains that scalar
        list_of_discounted_returns = rewards.size*[sum_of_discounted_rewards]

        return list_of_discounted_returns

    def _discounted_cumsum(self, rewards):
        """
            Input:
                a list of length T 
                a list of rewards {r_0, r_1, ..., r_t', ... r_{T-1}} from a single rollout of length T
            Output: 
                a list of length T
                a list where the entry in each index t is sum_{t'=t}^{T-1} gamma^(t'-t) * r_{t'}
        """

        all_discounted_cumsums = []

        # for loop over steps (t) of the given rollout
        for start_time_index in range(rewards.size): 

            # 1) create a list of indices (t'): goes from t to T-1
            indices = np.arange(start_time_index, rewards.size)

            # 2) create a list where the entry at each index (t') is gamma^(t'-t)
            discounts = np.power(self.gamma, indices-start_time_index)

            # 3) create a list where the entry at each index (t') is gamma^(t'-t) * r_{t'}
            # Hint: remember that t' goes from t to T-1, so you should use the rewards from those indices as well
            discounted_rtg = discounts * rewards[start_time_index:]

            # 4) calculate a scalar: sum_{t'=t}^{T-1} gamma^(t'-t) * r_{t'}
            sum_discounted_rtg = sum(discounted_rtg)

            # appending each of these calculated sums into the list to return
            all_discounted_cumsums.append(sum_discounted_rtg)
        list_of_discounted_cumsums = np.array(all_discounted_cumsums)
        return list_of_discounted_cumsums 