Berkeley CS285 HW3
==================
# This document does not fully satisfy the elements required in the instruction. It is just for self-organization purpose. Also, arguments and codes are modified.
## Question 1
### Basic Q-learning performance. (DQN)

#### - Include a learning curve plot showing the performance of your implementation on the game Pong. The x-axis should correspond to number of time steps (consider using scientific notation) and the y-axis should show the mean 100-episode reward as well as the best mean reward.

![Alt text](./pictures/Q1_AverageReturn.png "Average Return of Pong Game with DQN")
![Alt text](./pictures/Q1_BestReturn.png "Best Return of Pong Game with DQN")

This pong environment takes images of 4 time history as observation and handles with CNN network. In this run, I trained 5000000 time steps in total and the exploration rate was linearly decreased from 1 to 0.1 as the time step increased from 0 to 1000000 and 0.1 to 0.01 as time step increased from 1000000 to 5000000. It achieved around 17 to 18 average return.

---------------------------------------
## Question 2
### Double Q-learning (DDQN)
#### Use the double estimator to improve the accuracy of your learned Q values. This amounts to using the online Q network (instead of the target Q network) to select the best action when computing target values. Compare the performance of DDQN to vanilla DQN.

![Alt text](./pictures/Q2_DoubleQEffectCompare.png "Comparison Between Vanilla Q-Learning and Double Q-Learning")

Each Vallina Q-learning and double Q-learning algorithm was run with 3 seeds. As It can be seen from the figure, all 3 run o f double Q-learning achieved better performance than vallina Q-learning. It it known that since vallina Q-learning overestimate the value from the max operation, decoupling the action selection network and value estimation network alleviate the overestimation problem.

---------------------------------------
## Question 3
### Hyperparameter Tuning 
#### Choose one hyperparameterof your choice and run at least three other settings of this hyperparameter, in addition to the one used in Question 1, and plot all four values on the same graph.

![Alt text](./pictures/Q3_BatchSizeEffectCompare.png "Effect of Batch Size Comparision")

The hyperparameter I chose is the batch size. In addition to the original batch size of 32, which was run in Quesion 1, batch size of 16, 64, 128 was chosen. Batch size of 32 achieved the highest performance, but It seems reasonable to tune the learning rate also, since change of batch size affect the sensitivity of learning rate. 

---------------------------------------

