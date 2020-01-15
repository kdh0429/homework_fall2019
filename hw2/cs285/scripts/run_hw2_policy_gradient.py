import os
import time

from cs285.infrastructure.rl_trainer import RL_Trainer
from cs285.agents.pg_agent import PGAgent

class PG_Trainer(object):

    def __init__(self, params):
    
        #####################
        ## SET AGENT PARAMS
        #####################

        computation_graph_args = {
            'n_layers': params['n_layers'],
            'size': params['size'],
            'learning_rate': params['learning_rate'],
        }
        estimate_advantage_args = {
            'gamma': params['discount'],
            'standardize_advantages': params['standardize_advantages'],
            'reward_to_go': params['reward_to_go'],
            'nn_baseline': params['nn_baseline'],
            'gae': params['gae'],
            'gae_gamma': params['gae_gamma'],
            'gae_lambda': params['gae_lambda']
        }

        train_args = {
            'num_agent_train_steps_per_iter': params['num_agent_train_steps_per_iter'],
        }

        agent_params = {**computation_graph_args, **estimate_advantage_args, **train_args}

        self.params = params
        self.params['agent_class'] = PGAgent
        self.params['agent_params'] = agent_params
        self.params['batch_size_initial'] = self.params['batch_size']

        ################
        ## RL TRAINER
        ################

        self.rl_trainer = RL_Trainer(self.params)

    def run_training_loop(self):

        self.rl_trainer.run_training_loop(
            self.params['n_iter'], 
            collect_policy = self.rl_trainer.agent.actor,
            eval_policy = self.rl_trainer.agent.actor,
            )
        if self.params['render_after_training'] == 1:
            self.rl_trainer.eval_render(self.rl_trainer.agent.actor)

    def load_trained_agent_render(self):
        self.rl_trainer.agent.actor.restore('/home/kim/cs285_ws/homework_fall2019/hw2/cs285/data/pg_todo_CartPole-v0_15-01-2020_15-42-29/policy_itr_99')
        self.rl_trainer.eval_render(self.rl_trainer.agent.actor)


def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='todo')
    parser.add_argument('--n_iter', '-n', type=int, default=200)
        
    parser.add_argument('--reward_to_go', '-rtg', type=int, default=0)
    parser.add_argument('--nn_baseline', type=int, default=0)
    parser.add_argument('--standardize_advantages', '-sa', type=int, default=0)
    parser.add_argument('--batch_size', '-b', type=int, default=1000) #steps collected per train iteration

    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=64)

    parser.add_argument('--ep_len', type=int) #students shouldn't change this away from env's default
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--use_gpu', '-gpu', type=int, default=1)
    parser.add_argument('--which_gpu', '-gpu_id', default=0)
    parser.add_argument('--video_log_freq', type=int, default=-1)   # video log disabled
    parser.add_argument('--scalar_log_freq', type=int, default=1)

    parser.add_argument('--use_wandb', type=int, default=0)
    parser.add_argument('--n_eval', type=int, default=5)
    parser.add_argument('--render_after_training', type=int, default=0)
    parser.add_argument('--n_worker', type=int, default=1)
    parser.add_argument('--gae', type=int, default= 0)
    parser.add_argument('--gae_gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.96)

    parser.add_argument('--load_existing_agent', type=int, default=0)

    args = parser.parse_args()

    # convert to dictionary
    params = vars(args)
    
    # for this assignment, we train on everything we recently collected
    # so making train_batch_size=batch_size 
    params['train_batch_size']=params['batch_size']

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    logdir_prefix = 'pg_'

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = logdir_prefix + args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    ###################
    ### RUN TRAINING
    ###################

    trainer = PG_Trainer(params)
    if not params['load_existing_agent']:
        trainer.run_training_loop()
    else:
        trainer.load_trained_agent_render()


if __name__ == "__main__":
    main()
