Berkeley CS285 HW2
==================
# This document does not fully satisfy the elements required in the instruction. It is just for self-organization purpose.
## Problem 3
### * Create two graphs:

#### - In the first graph, compare the learning curves (average return at each iteration)
for the experiments prefixed with sb_. (The small batch experiments.)
![Alt text](./pictures/small_batch.png "Small Batch Training Curve")
dainty-wave-8 : python run_hw2_policy_gradient.py --env_name CartPole-v0 -n 100 -b 1000 -dsa --n_layers 1 --size 32 --use_wandb 1 --n_worker 1 --render_after_training 0 --exp_name sb_no_rtg_dsa
curious-pyramid-9 : python run_hw2_policy_gradient.py --env_name CartPole-v0 -n 100 -b 1000 -rtg -dsa --n_layers 1 --size 32 --use_wandb 1 --n_worker 1 --render_after_training 0 --exp_name sb_rtg_dsa
rare-sunset-10 : python run_hw2_policy_gradient.py --env_name CartPole-v0 -n 100 -b 1000 --n_layers 1 --size 32 --use_wandb 1 --n_worker 1 --render_after_training 0 --exp_name sb_rtg_na
• -n : Number of iterations.
• -b : Batch size (number of state-action pairs sampled while acting according to the
current policy at each iteration).
• -dsa : Flag: if present, sets standardize_advantages to False. Otherwise, by
default, standardize_advantages=True.
• -rtg : Flag: if present, sets reward_to_go=True. Otherwise, reward_to_go=False
by default.
• --exp_name : Name for experiment, which goes into the name for the data logging
directory.

#### – In the second graph, compare the learning curves for the experiments prefixed
with lb_. (The large batch experiments.)
![Alt text](./pictures/large_batch.png "Large Batch Training Curve")
drawn-disco-11 : python run_hw2_policy_gradient.py --env_name CartPole-v0 -n 100 -b 5000 -dsa --n_layers 1 --size 32 --use_wandb 1 --n_worker 1 --render_after_training 0 --exp_name lb_no_rtg_dsa
colorful-microwave-12 : python rrun_hw2_policy_gradient.py --env_name CartPole-v0 -n 100 -b 5000 -rtg -dsa --n_layers 1 --size 32 --use_wandb 1 --n_worker 1 --render_after_training 0 --exp_name lb_rtg_dsa
true-water-13 : python run_hw2_policy_gradient.py --env_name CartPole-v0 -n 100 -b 5000 --n_layers 1 --size 32 --use_wandb 1 --n_worker 1 --render_after_training 0 --exp_name lb_rtg_na
• -n : Number of iterations.
• -b : Batch size (number of state-action pairs sampled while acting according to the
current policy at each iteration).
• -dsa : Flag: if present, sets standardize_advantages to False. Otherwise, by
default, standardize_advantages=True.
• -rtg : Flag: if present, sets reward_to_go=True. Otherwise, reward_to_go=False
by default.
• --exp_name : Name for experiment, which goes into the name for the data logging
directory.

### * Answer the following questions briefly:
– Which value estimator has better performance without advantage-standardization:
the trajectory-centric one, or the one using reward-to-go?
  A : Reward-to-go
– Did advantage standardization help?
  A : No. Actually for small batch experiment, standardizing advantage degraded the performance.
– Did the batch size make an impact?
  A : Yes, larger batch size improved the performance.
---------------------------------------
The result of behavior cloning with 'Ant-v2' environent is shown as above. The only change from default setting is that 'num_agent_train_steps_per_iter' is set as 2000.
Average return of expert is 4713.653(Initial_DataCollection_AverageReturn) and the policy trained with behavior cloning accomplishes average return of 3613.055 with standand devidation of 1395.815(from 5 evaluation episodes).

![Alt text](./pictures/Humanoid_BC.png "Behavior Cloning of Humanoid")

The result of behavior cloning with 'Humanoid-v2' environent is shown as above. Size of MLP is changed from 64 to 128 compared to 'Ant-v2' environment, considering higher dimension of observation and action space. 'num_agent_train_steps_per_iter' is set as 2000 as before. 
Average return of expert is 10344.518(Initial_DataCollection_AverageReturn), but the policy trained with behavior cloning only acheives 299.28 with 30.846 standard deviation. 


### 3. Experiment with one set of hyperparameter that affects the performance of the behavioral cloning agent, such as the number of demonstrations, the number of training epochs, the variance of the expert policy, or something that you come up with yourself. For one of the tasks used in the previous question, show a graph of how the BC agent’s performance varies with the value of this hyperparameter, and state the hyperparameter and a brief rationale for why you chose it in the caption for the graph.


![Alt text](./pictures/Retrun_Iteration.jpg "Mean and Standard Deviation of Return with Training Iteration")
Figure above is the mean and standard deviation of ant behavior cloning agent return as training iteration changes. As the training iteration changes from 1000 to 2400, return tends to increase. At iteration 2400, the mean return is peak and std is lowest. As iteration increases from 2400, it seems the agent has been overfitted to training policy.

## Section 2

### 2. Run DAgger and report results on one task in which DAgger can learn a better policy than behavioral cloning. Report your results in the form of a learning curve, plotting the number of DAgger iterations vs. the policy’s mean return, with error bars to show the standard deviation. Include the performance of the expert policy and the behavioral cloning agent on the same plot. In the caption, state which task you used, and any details regarding network architecture, amount of data, etc. (as in the previous section).


![Alt text](./pictures/Humanoid_DAgger.png "Return of DAgger and Expert Policy")
Humanoid policy trained with DAgger algorithm is multi later fully connected policy with 2 hidden layers and 64 hidden neurons. Return of expert policy is above 10000 and DAgger policy achieves maximum of around 8000. DAgger preety improves performance compared to simple behavior cloning by sampling trajectory from trained policy then relabelling from expert so that the training data distribution matches test data distribution.
