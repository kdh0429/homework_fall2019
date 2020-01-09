import time

from collections import OrderedDict
import pickle
import numpy as np
import tensorflow as tf
import gym
import os
import wandb

from cs285.infrastructure.utils import *
from cs285.infrastructure.tf_utils import create_tf_session

# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40 # we overwrite this in the code below

class RL_Trainer(object):

    def __init__(self, params):

        #############
        ## INIT
        #############

        # Get params, create logger, create TF session
        self.params = params
        self.sess = create_tf_session(self.params['use_gpu'], which_gpu=self.params['which_gpu'])

        # Set random seeds
        seed = self.params['seed']
        tf.set_random_seed(seed)
        np.random.seed(seed)

        #############
        ## ENV
        #############

        # Make the gym environment
        self.env = gym.make(self.params['env_name'])
        self.env.seed(seed)

        # Maximum length for episodes
        self.params['ep_len'] = self.params['ep_len'] or self.env.spec.max_episode_steps

        # Is this env continuous, or self.discrete?
        discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        self.params['agent_params']['discrete'] = discrete

        # Observation and action sizes
        ob_dim = self.env.observation_space.shape[0]
        ac_dim = self.env.action_space.n if discrete else self.env.action_space.shape[0]
        self.params['agent_params']['ac_dim'] = ac_dim
        self.params['agent_params']['ob_dim'] = ob_dim
        print("******************************************************************")
        print("Action Dimension: ", self.params['agent_params']['ac_dim'])
        print("Observation Dimension: ", self.params['agent_params']['ob_dim'])
        print("******************************************************************")

        if self.params['use_wandb'] == 1:
            wandb.init(project="cs285_hw2", tensorboard=False)
            wandb.config.env_name = self.params['env_name']
            wandb.config.exp_name = self.params['exp_name']
            wandb.config.ep_len = self.params['ep_len']
            wandb.config.num_agent_train_steps_per_iter = self.params['num_agent_train_steps_per_iter']
            wandb.config.n_iter = self.params['n_iter']
            wandb.config.batch_size = self.params['batch_size']
            wandb.config.eval_batch_size = self.params['eval_batch_size']
            wandb.config.train_batch_size = self.params['train_batch_size']
            wandb.config.n_layers = self.params['n_layers']
            wandb.config.size = self.params['size']
            wandb.config.learning_rate = self.params['learning_rate']
            wandb.config.scalar_log_freq = self.params['scalar_log_freq']
            wandb.config.use_gpu = self.params['use_gpu']
            wandb.config.which_gpu = self.params['which_gpu']
            wandb.config.seed = self.params['seed']
            wandb.config.n_eval = self.params['n_eval']

        #############
        ## AGENT
        #############

        agent_class = self.params['agent_class']
        self.agent = agent_class(self.sess, self.env, self.params['agent_params'])

        #############
        ## INIT VARS
        #############

        tf.global_variables_initializer().run(session=self.sess)

    def run_training_loop(self, n_iter, collect_policy, eval_policy,
                        initial_expertdata=None, relabel_with_expert=False,
                        start_relabel_with_expert=1, expert_policy=None):
        """
        :param n_iter:  number of (dagger) iterations
        :param collect_policy:
        :param eval_policy:
        :param initial_expertdata:
        :param relabel_with_expert:  whether to perform dagger
        :param start_relabel_with_expert: iteration at which to start relabel with expert
        :param expert_policy:
        """

        # init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()

        for itr in range(n_iter):
            print("\n\n********** Iteration %i ************"%itr)

            # decide if metrics should be logged
            if itr % self.params['scalar_log_freq'] == 0:
                self.log_metrics = True
            else:
                self.log_metrics = False

            # collect trajectories, to be used for training
            st = time.time()
            training_returns = self.collect_training_trajectories(itr,
                                initial_expertdata, collect_policy,
                                self.params['batch_size'])
            print("Sameple Time: ", time.time()- st)
            paths, envsteps_this_batch = training_returns
            self.total_envsteps += envsteps_this_batch

            # relabel the collected obs with actions from a provided expert policy
            if relabel_with_expert and itr>=start_relabel_with_expert:
                paths = self.do_relabel_with_expert(expert_policy, paths)

            # add collected data to replay buffer
            self.agent.add_to_replay_buffer(paths)

            # train agent (using sampled data from replay buffer)
            self.train_agent()

            # log/save
            if self.log_metrics:

                # perform logging
                print('\nBeginning logging procedure...')
                self.perform_logging(itr, paths, eval_policy)


                if self.params['save_params']:
                    # save policy
                    print('\nSaving agent\'s actor...')
                    self.agent.actor.save(self.params['logdir'] + '/policy_itr_'+str(itr))

    ####################################
    ####################################

    def collect_training_trajectories(self, itr, load_initial_expertdata, collect_policy, batch_size):
        # TODO: GETTHIS from HW1
        if itr == 0 and load_initial_expertdata:
            with open(load_initial_expertdata,'rb') as f:
                paths = pickle.load(f)
            envsteps_this_batch = 0
            return paths, envsteps_this_batch

        # TODO collect data to be used for training
        # HINT1: use sample_trajectories from utils
        # HINT2: you want each of these collected rollouts to be of length self.params['ep_len']
        print("\nCollecting data to be used for training...")
        paths, envsteps_this_batch = sample_trajectories(self.env, collect_policy, batch_size, self.params['ep_len'])

        return paths, envsteps_this_batch

    def train_agent(self):
        # TODO: GETTHIS from HW1
        print('\nTraining agent using sampled data from replay buffer...')
        for train_step in range(self.params['num_agent_train_steps_per_iter']):

            # TODO sample some data from the data buffer
            # HINT1: use the agent's sample function
            # HINT2: how much data = self.params['train_batch_size']
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.agent.sample(self.params['train_batch_size'])
            # TODO use the sampled data for training
            # HINT: use the agent's train function
            # HINT: print or plot the loss for debugging!
            self.agent.train(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)

    def do_relabel_with_expert(self, expert_policy, paths):
        # TODO: GETTHIS from HW1 (although you don't actually need it for this homework)
        print("\nRelabelling collected observations with labels from an expert policy...")

        # TODO relabel collected obsevations (from our policy) with labels from an expert policy
        # HINT: query the policy (using the get_action function) with paths[i]["observation"]
        # and replace paths[i]["action"] with these expert labels
        for path in paths:
            path["action"] = expert_policy.get_action(path["observation"])
        return paths

    ####################################
    ####################################

    def perform_logging(self, itr, paths, eval_policy):

        # collect eval trajectories, for logging
        print("\nCollecting data for eval...")
        eval_paths = sample_n_trajectories(self.env, eval_policy, self.params['n_eval'], self.params['ep_len'])

        # save eval metrics
        if self.log_metrics:
            # returns, for logging
            train_returns = [path["reward"].sum() for path in paths]
            eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

            # episode lengths, for logging
            train_ep_lens = [len(path["reward"]) for path in paths]
            eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

            # decide what to log
            logs = OrderedDict()
            logs["Eval_AverageReturn"] = np.mean(eval_returns)
            logs["Eval_StdReturn"] = np.std(eval_returns)
            logs["Eval_MaxReturn"] = np.max(eval_returns)
            logs["Eval_MinReturn"] = np.min(eval_returns)
            logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

            logs["Train_AverageReturn"] = np.mean(train_returns)
            logs["Train_StdReturn"] = np.std(train_returns)
            logs["Train_MaxReturn"] = np.max(train_returns)
            logs["Train_MinReturn"] = np.min(train_returns)
            logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

            logs["Train_EnvstepsSoFar"] = self.total_envsteps
            logs["TimeSinceStart"] = time.time() - self.start_time


            if itr == 0:
                self.initial_return = np.mean(train_returns)
            logs["Initial_DataCollection_AverageReturn"] = self.initial_return

            if self.params['use_wandb'] == 1:
                wandb.log(logs)

    def eval_render(self,eval_policy):
        ob = self.env.reset() # HINT: should be the output of resetting the env
        step = 0
        while True:
            self.env.render()
            ac = eval_policy.get_action(ob) # HINT: query the policy's get_action function
            ac = ac[0]
            ob, rew, done, _ = self.env.step(ac)
            step += 1
            if done or step > self.params['ep_len']:
                step = 0
                ob = self.env.reset() # HINT: should be the output of resetting the env
            