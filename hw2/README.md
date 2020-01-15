Berkeley CS285 HW2
==================
# This document does not fully satisfy the elements required in the instruction. It is just for self-organization purpose. Also, arguments and codes are modified.
## Problem 3
### Create two graphs:

#### - In the first graph, compare the learning curves (average return at each iteration) for the experiments prefixed with sb_. (The small batch experiments.)
![Alt text](./pictures/CartPole_SB.png "Small Batch Validation Curve")

deft-pine-91 : python run_hw2_policy_gradient.py --env_name CartPole-v0 -n 100 -b 1000 -sa 0 -l 1 -s 32 --use_wandb 1 --render_after_training 0 --n_worker 1 --gae 0

daily-planet-92 : python run_hw2_policy_gradient.py --env_name CartPole-v0 -n 100 -b 1000 -sa 0 -rtg 1 -l 1 -s 32 --use_wandb 1 --render_after_training 0 --n_worker 1 --gae 0

good-lion-93 : python run_hw2_policy_gradient.py --env_name CartPole-v0 -n 100 -b 1000 -sa 1 -rtg 1 -l 1 -s 32 --use_wandb 1 --render_after_training 0 --n_worker 1 --gae 0

• -n : Number of iterations.

• -b : Batch size (number of state-action pairs sampled while acting according to the
current policy at each iteration).

• -sa : Flag: standardize advantages

• -rtg : Flag: Sets reward_to_go


#### – In the second graph, compare the learning curves for the experiments prefixed with lb_. (The large batch experiments.)
![Alt text](./pictures/CartPole_LB.png "Large Batch Validation Curve")

clean-star-94 : python run_hw2_policy_gradient.py --env_name CartPole-v0 -n 100 -b 5000 -sa 0 -rtg 0 -l 1 -s 32 --use_wandb 1 --render_after_training 0 --n_worker 1 --gae 0

lunar-pond-95 : python run_hw2_policy_gradient.py --env_name CartPole-v0 -n 100 -b 5000 -sa 0 -rtg 1 -l 1 -s 32 --use_wandb 1 --render_after_training 0 --n_worker 1 --gae 0

colorful-bee-96 : python run_hw2_policy_gradient.py --env_name CartPole-v0 -n 100 -b 5000 -sa 1 -rtg 1 -l 1 -s 32 --use_wandb 1 --render_after_training 0 --n_worker 1 --gae 0

• -n : Number of iterations.

• -b : Batch size (number of state-action pairs sampled while acting according to the
current policy at each iteration).

• -sa : Flag: standardize advantages

• -rtg : Flag: Sets reward_to_go

### Answer the following questions briefly:

#### – Which value estimator has better performance without advantage-standardization: the trajectory-centric one, or the one using reward-to-go?

    A : Reward-to-go
  
#### – Did advantage standardization help?

    A : Not actually. The performance did not depend on whether to standardize advantage or not. 
  
#### – Did the batch size make an impact?

    A : Yes, larger batch size improved the performance.
---------------------------------------
## Problem 4
### InvertedPendulum:
#### Find the smallest batch size b* and largest learning rate r* that gets to optimum (maximum score of 1000) in less than 100 iterations

![Alt text](./pictures/InvertedPendulum.png "Evaluation Cureve of Inverted Pendulum with Batch Size 1000 and Learning Rate  0.015")

python run_hw2_policy_gradient.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 1000 -lr 0.015 -rtg 1 -sa 1 --use_wandb 1 --render_after_training 1 --n_worker 1 --exp_name ip_b1000_r0.015


---------------------------------------
## Problem 6
### LunarLander: 
#### Plot a learning curve for the above command. You should expect to achieve an average return of around 180.

![Alt text](./pictures/LunarLander_nnbaseline.png "Evaluation Cureve of LunarLander")

---------------------------------------
## Problem 7
### HalfCheetah: 
#### Provide a single plot with the learning curves for the HalfCheetah experiments over batch sizes b ∈ [10000, 30000, 50000] and learning rates r ∈ [0.005, 0.01, 0.02]. Also, describe in words how the batch size and learning rate affected task performance.

![Alt text](./pictures/HalfCheetahHyperParamCompare.png "Half Cheetah Hyper Parameter Search")

- misty-vortex-98 : batch size 10000, learning rate 0.02
- lucky-pyramid-99 : batch size 10000, learning rate 0.01
- happy-wildflower-100 : batch size 10000, learning rate 0.005
- royal-durian-101 : batch size 30000, learning rate 0.02
- sage-flower-102: batch size 30000, learning rate 0.01
- still-surf-103 : batch size 30000, learning rate 0.005
- deft-paper-104 : batch size 50000, learning rate 0.02
- genial-shape-105 : batch size 50000, learning rate 0.01
- absurd-eon-106 : batch size 50000, learning rate 0.005

In general, larget batch size and higher learning rate accompished better performance. Especially, larget batch helps. The best performance was accompished with batch size 50000 and learning rate 0.01

#### Provide a single plot with the learning curves for four runs below. The run with both reward-to-go and the baseline should achieve an average score close to 200.


![Alt text](./pictures/HalfCheetah_rtg_nnbaseline_Compare.png "Half Cheetah Reward to Go and NN Baseline Compare")

• olive-salad-81 : without reward to go and without neural network baseline

    python run_hw2_policy_gradient.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b 50000 -lr 0.02 -rtg 0 --nn_baseline 0 --use_wandb 1 --gae 0 --render_after_training 0 --n_worker 1 --exp_name hc_b50000_lr0.02

• mild-galaxy-82 : with reward to go and without neural network baseline

    python run_hw2_policy_gradient.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b 50000 -lr 0.02 -rtg 1 --nn_baseline 0 --use_wandb 1 --dont_gae --render_after_training 0 --n_worker 1 --exp_name hc_b50000_lr0.02

• pleasant-meadow-83 : without reward to go and with neural network baseline

    python run_hw2_policy_gradient.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b 50000 -lr 0.02 -rtg 0 --nn_baseline 1 --use_wandb 1 --gae 0 --render_after_training 0 --n_worker 1 --exp_name hc_b50000_lr0.02

• crisp-cloud-84: with reward to go and with neural network baseline

    python run_hw2_policy_gradient.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b 50000 -lr 0.02 -rtg 1 --nn_baseline 1--use_wandb 1 --dont_gae --render_after_training 0 --n_worker 1 --exp_name hc_b50000_lr0.02


---------------------------------------
## Bonus!
### A serious bottleneck in the learning, for more complex environments, is the samplecollection time. In infrastructure/rl trainer.py, we only collect trajectories in a single thread, but this process can be fully parallelized across threads to get a useful speedup. Implement the parallelization and report on the difference in training time.

![Alt text](./pictures/ParallelizationTimeCompare.png "Time Since Start Comparision")

The comparision was held with LunarLander environment handled in problem 6. MultiProccesing module was used to parralelize sampling reffering to OpenAI baseline SubprocVecEnv. 16 independent enviroments with workers are created parralelly thus shorten sampling time.

### Implement GAE-λ for advantage estimation. 1 Run experiments in a MuJoCo gym environment to explore whether this speeds up training. (Walker2d-v1 may be good for this.)

![Alt text](./pictures/Walker2D_GAE_Compare.png "Walker2d GAE Comparision")

Walker2d-v2 environment was used for comparision. gamma and lambda for GAE was set to 0.99 and 0.96 each. 2 hidden layers with 64 hidden neurons per layer was used for value function estimation. Learning rate was set as 0.005, with 50000 batch size. By using GAE, return has increase, but the variance is large. I am not sure whether the implementation is correct. It should be checked.
