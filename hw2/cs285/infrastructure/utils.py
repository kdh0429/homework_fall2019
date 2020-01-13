import numpy as np
import time
from cs285.infrastructure.multiprocessing_env import SubprocVecEnv
import gym
############################################
############################################

def sample_trajectory(env, policy, max_path_length, num_worker=1, render=False, render_mode=('rgb_array')):

    # initialize env for the beginning of a new rollout
    ob = env.reset()  # TODO: GETTHIS from HW1
    
    # init vars
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    for i in range(num_worker):
        obs.append([])
        acs.append([])
        rewards.append([])
        next_obs.append([])
        terminals.append([])

    steps = 0
    rollout_done = [False for _ in range(num_worker)]

    while True:
        # use the most recent ob to decide what to do
        ac = policy.get_action(ob) # TODO: GETTHIS from HW1
        for i in range(num_worker):
            obs[i].append(ob[i])
            acs[i].append(ac[i])
        #ac = ac[0]
        #acs.append(ac)

        # take that action and record results
        ob, rew, done, _ = env.step(ac)

        # record result of taking that action
        steps += 1

        # End the rollout if the rollout ended 
        # Note that the rollout can end due to done, or due to max_path_length
        
        for i in range(num_worker):
            next_obs[i].append(ob[i])
            rewards[i].append(rew[i])
            terminals[i].append(done[i] or (steps > max_path_length))
            if rollout_done[i] is False:
                rollout_done[i] = done[i] or (steps > max_path_length) # TODO: GETTHIS from HW1
                

        obs_trajectories, acs_trajectories, rewards_trajectories, next_obs_trajectories, terminals_trajectories, image_obs_trajectories = [], [], [], [], [], []

        if all(rollout_done):
            for i in range(num_worker):
                for j in range(len(obs[i])):
                    obs_trajectories.append(obs[i][j])
                for j in range(len(acs[i])):
                    acs_trajectories.append(acs[i][j])
                for j in range(len(rewards[i])):
                    rewards_trajectories.append(rewards[i][j])
                for j in range(len(next_obs[i])):
                    next_obs_trajectories.append(next_obs[i][j])
                for j in range(len(terminals[i])):
                    terminals_trajectories.append(terminals[i][j])    
            break
        
    return Path(obs_trajectories, image_obs_trajectories, acs_trajectories, rewards_trajectories, next_obs_trajectories, terminals_trajectories)

def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, num_worker=1, render=False, render_mode=('rgb_array'), ):

    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:
        this_batch_path = sample_trajectory(env, policy, max_path_length, num_worker)
        paths.append(this_batch_path)
        timesteps_this_batch += get_pathlength(this_batch_path)

    return paths, timesteps_this_batch

def sample_n_trajectories(env, policy, ntraj, max_path_length, num_worker=1, render=False, render_mode=('rgb_array')):
    
    paths = []
    for num_traj in range(ntraj):
        paths.append(sample_trajectory(env, policy, max_path_length, num_worker, render, render_mode))
        
    return paths

############################################
############################################

def Path(obs, image_obs, acs, rewards, next_obs, terminals):
    """
        Take info (separate arrays) from a single rollout
        and return it in a single dictionary
    """
    if len(image_obs) > 0:
        image_obs = np.stack(image_obs, axis=0)
    return {"observation" : np.array(obs, dtype=np.float32),
            "image_obs" : np.array(image_obs, dtype=np.uint8),
            "reward" : np.array(rewards, dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)}


def convert_listofrollouts(paths):
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    concatenated_rewards = np.concatenate([path["reward"] for path in paths])
    unconcatenated_rewards = [path["reward"] for path in paths]
    return observations, actions, next_observations, terminals, concatenated_rewards, unconcatenated_rewards


def get_pathlength(path):
    return len(path["reward"])

############################################
############################################

def env_generator(env_name, seed):
    """Return env creating function (with normalizers)."""

    def _thunk(rank):
        env = gym.make(env_name)
        env.seed(seed + rank)
        return env

    return _thunk


def make_envs(env_gen, n_envs):
    """Make multiple environments running on multiprocssors."""
    envs = [env_gen(i) for i in range(n_envs)]
    subproc_env = SubprocVecEnv(envs)
    return subproc_env