# Isaac-PhyRL-Go2

## To Do ##
* [x] Add BEV map to the repo
* [ ] Fast Marching Method (FMM) implementation
* [ ] Go2 real robot deployment
  * [ ] Gazebo real-time testing
  * [ ] ROS/ROS2 encapsulation
* [ ] Incorporate more challenging scenarios
  * [ ] Dense forests (sandy terrain, trees)
  * [ ] inclined staircases, and rainy conditions
* [ ] Restructure the code as FSM and add teleoperation (optional)
* [ ] Migration to Isaac-Lab

## User Guide

### Dependencies
* *Python - 3.8 above*
* *PyTorch - 1.10.0*
* *Isaac Gym - Preview 4*

### Setup

1. Clone this repository:
```bash
git clone git@github.com:Charlescai123/isaac-phyrl-go2.git
```

2. Create the conda environment with:
```bash
conda env create -f environment.yml
```

3. Activate conda environment and Install `rsl_rl` lib:
```bash
conda activate phyrl-go2
cd extern/rsl_rl && pip install -e .
```

4. Download and install IsaacGym:
* Download [IsaacGym](https://developer.nvidia.com/isaac-gym) and extract the downloaded file to the root folder.
* navigate to the `isaacgym/python` folder and install it with commands:
* ```bash
  cd isaacgym/python && pip install -e .
  ```
* Test the given example (ignore the error about GPU not being utilized if any): 
* ```bash
  cd examples && python 1080_balls_of_solitude.py
  ```

## 🎉 Acknowledgments

Special thanks to the contributions from these repos:

- [rsl\_rl](https://github.com/leggedrobotics/rsl_rl.git): Template for standard Reinforcement Learning algorithms.
- [legged\_gym](https://github.com/leggedrobotics/legged_gym): Collection of simulation environments for legged robots.
- [cajun](https://github.com/yxyang/cajun): Some baseline code for MPC-based control of legged robots.