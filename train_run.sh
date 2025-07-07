#!/bin/bash
#### CuRLA ####
# python train.py -start_carla --model_name hc_v6 --reward_fn highway_reward

#### One-Fold CL ####
# python train.py -start_carla --model_name hc_v4 --reward_fn highway_reward

#### SCA ####
# python train.py -start_carla --model_name sca --reward_fn reward_speed_centering_angle_multiply  
