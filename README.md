# CuRLA

## About the Project
This project is the supporting codebase to the [CuRLA](https://www.scitepress.org/Papers/2025/131470/131470.pdf) paper.

We have used the urban driving simulator [CARLA](http://carla.org/) (version 0.9.5) as our environment.

## Contributions

- We provide a reward function (highway_reward) that optimizes the average speed metric of the vehicle.
- Inclusion of jerk and collision penalties in the reward functions to incorporate a safety metric in the reward itself.
- Addition of traffic to enable a curriculum learning methodology for training.

## Related Work

This project builds upon the contributions of [Deep RL for Autonomous Driving](https://github.com/bitsauce/Carla-ppo) for the VAE+PPO architecture, as well as the custom route and lap environments in CARLA.

# How to Run

## Prerequisites

- Python 3.8
- [CARLA 0.9.5](https://github.com/carla-simulator/carla/tree/0.9.5) (may also work with later versions)
    - Our code expects the CARLA python API to be installed and available through `import carla` (see [this](https://carla.readthedocs.io/en/latest/build_system/#pythonapi))
    - We also recommend building a editor-less version of Carla by running the `make package` command in the root directory of CARLA.
    - Note that the map we use, `Town07`, may not be included by default when running `make package`. Add `+MapsToCook=(FilePath="/Game/Carla/Maps/Town07")` to `Unreal/CarlaUE4/Config/DefaultGame.ini` before running `make package` to solve this.
- [TensorFlow for GPU](https://www.tensorflow.org/) (we have used version 1.13, may work with later versions)
- [OpenAI gym](https://github.com/openai/gym) (we used version 0.12.0)
- [OpenCV for Python](https://pypi.org/project/opencv-python/) (we used version 4.0.0)
- A GPU with at least 4 GB VRAM (we used a GeForce GTX 3080)

## Running a Trained Agent
To (further) train one of the pretrained agents referred to in the CuRLA paper, uncomment the relevant line in the [train_run.sh](https://github.com/Anjel-Patel/AutoLaneChange/blob/master/train_run.sh) file and then run it in a conda environment where you have the installed prerequisites.

To remove traffic during training, comment lines 156-165 in the [CarlaEnv/carla_lap_env.py](https://github.com/Anjel-Patel/AutoLaneChange/blob/master/CarlaEnv/carla_lap_env.py) file.

# File Overview

| File                         | Description                                                                                                                     |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| train.py                     | Script for training a PPO agent in the lap environment                                                                          |
| run_eval.py                  | Script for running a trained model in eval mode                                                                                 |
| utils.py                     | Contains various mathematical, tensorflow, DRL utility functions                                                                |
| ppo.py                       | Contains code for constructing the PPO model                                                                                    |
| reward_functions.py          | Contains all reward functions                                                                                                   |
| vae_common.py                | Contains functions related to VAE loading and state encoding                                                                    |
| inspect_agent.py             | Script used to inspect the behavior of the agent as the VAE's latent space vector z is annealed                                 |
| models/                      | Folder containing agent checkpoints, tensorboard log files, and video recordings                                                |
| doc/                         | Folder containing figures that are used in this readme, in addition to a PDF version of the project write-up                    |
| vae/                         | Folder containing variational autoencoder related code                                                                          |
| vae/train_vae.py             | Script for training a variational autoencoder                                                                                   |
| vae/models.py                | Contains code for constructing MLP and CNN-based VAE models                                                                     |
| vae/inspect_vae.py           | Script used to inspect how latent space vector z affects the reconstructions of a trained VAE                                   |
| vae/data/                    | Folder containing the images that were used when training the VAE model bundled with the repo                                   |
| vae/models/                  | Folder containing VAE model checkpoints and tensorboard logs                                                                    |
| CarlaEnv/                    | Folder containing code related to the CARLA environments                                                                        |
| CarlaEnv/carla_lap_env.py    | Contains code for the CarlaLapEnv class                                                                                         |
| CarlaEnv/carla_route_env.py  | Contains code for the CarlaRouteEnv class                                                                                       |
| CarlaEnv/collect_data.py     | Script used to manually drive a car in the environment to collect images that can be used to train a VAE                        |
| CarlaEnv/hud.py              | Code for the HUD displayed on the left-hand side of the spectating window                                                       |
| CarlaEnv/planner.py          | Global route planner used to find routes from A to B. Copied and modified from CARLA 0.9.4's PythonAPI                          |
| CarlaEnv/wrappers.py         | Contains wrapper classes for several CARLA classes                                                                              |
| CarlaEnv/agents/             | Contains code used by the route planner                                                                                         |


# Future Work

Here are a couple of ideas of how our work can be expanded or improved on:

- Multi-Objective Reinforcement Learning (MORL)
- Architecture revamp (incorporate Decision/Trajectory transformers)
