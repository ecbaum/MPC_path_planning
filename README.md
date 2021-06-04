# MPC_path_planning

The goal of this project is to use model predictive control (MPC) to plan vehicle trajectories that minimizes the some quantity such as distance or energy usage while following the constaints of the enviroment at the same time. The environemntal constraints will be modelled using potential fields as oppoosed to hard state constraints. The value function should be automatically generated using an image of a map and given an inital and final position, an optimal trajectory should be generated. If successful next step is to expand to 3D such that UAV trajectories cand be planned. 

Structure of the project is as following

1. Get familiar with CasADi and implmenet a basic nonlinear MPC problem to get familiar with how to work with in the CasADi framwork.

2. Use a motion model that can represent a reasonable vehicle trajectory like constant velocity or coordinate turn model. Generate trajectory given some constraints, initial state and a target state set

3. Expand the value function by including nonlinear potential field functions such as pointwise repulsive potential function

4. Generate automatic potential fields given a map of an evironment

5. Expand to 3D
