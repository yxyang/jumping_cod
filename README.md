# CAJun: Continuous Adaptive Jumping using a Learned Centroidal Controller
![](https://github.com/yxyang/cajun/blob/master/media/teaser.gif)

This repository contains the code for the paper ["CAJun: Continuous Adaptive Jumping using a Learned Centroidal Controller"](https://arxiv.org/abs/2306.09557).

The main contents of this repository include:

* The simulation environment and training code to reproduce the paper results.
* The real-robot interface to deploy the trained policy to a real-world Go1 quadrupedal robot.
* An Isaacgym implementation of the [Centroidal QP Controller](https://arxiv.org/abs/2009.10019), which can be executed efficiently in parallel in GPU.

## Reproducing Paper Results
### Setup the environment

First, make sure the environment is setup by following the steps in the [Setup](#Setup) section.

### Evaluating Policy

```bash
python -m src.agents.ppo.eval --logdir=example_checkpoints/bound_cajun/ --num_envs=1 --use_gpu=False --show_gui=True --use_real_robot=False --save_traj=False
```


## Usage

### Train Policies:
TODO



### Dog Tracer
![](https://github.com/yxyang/cajun/blob/master/media/dog_tracer.gif)
We provide a simple tool to visualize the logged robot trajectories. When evaluating PPO trajectories using `src.agents.ppo.eval` and set `save_traj=True`, the logged trajectory can be visualized using the `dog_tracer` web GUI.

To start `dog_tracer`, run:
```bash
python -m src.dog_tracer.dog_tracer
```
and load the trajectories from the UI.

## Setup
### Software
1. Create a new virtual environment under Python 3.6, 3.7, 3.8 (3.8 recommended).

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    Note that the `numpy` version must be no later than `1.19.5` (already specified in `requirements.txt`) to avoid conflict with the Isaac Gym utility files.

3. Download and install IsaacGym Preview 4:
    * Download IsaacGym from [https://developer.nvidia.com/isaac-gym](https://developer.nvidia.com/isaac-gym). Extract the downloaded file to the root folder.
    * `cd isaacgym/python && pip install -e .`
    * Try running example `cd examples && python 1080_balls_of_solitude.py`. The code is set to run on CPU so don't worry if you see an error about GPU not being utilized.

4. Install `rsl_rl` (adapted PPO implementation)
    ```bash
    cd rsl_rl && pip install -e .
    ```

5. Lastly, build and install the interface to Unitree's Go1 SDK. The Unitree [repo](https://github.com/unitreerobotics/unitree_legged_sdk) has been releasing new SDK versions. For convenience, we have included the version that we used in `third_party/unitree_legged_sdk`.

   * First, make sure the required packages are installed, following Unitree's [guide](https://github.com/unitreerobotics/unitree_legged_sdk). Most nostably, please make sure to install `Boost` and `LCM`:

   ```bash
   sudo apt install libboost-all-dev liblcm-dev
   ```

   * Then, go to `third_party/go1_sdk` and create a build folder:
   ```bash
   cd third_party/go1_sdk
   mkdir build && cd build
   ```

   Now, build the libraries and move them to the main directory by running:
   ```bash
   cmake ..
   make
   mv go1_interface* ../../..
   ```


### Robot Setup
Follow these steps if you want to run policies on the real robot.

1. **Disable Unitree's default controller**

    * By default, the Go1 robot enters `sport` mode and executes the default controller program at start-up.
To avoid interferences, make sure to disable Unitree's default controller before running any custom control code on the real robot.
    * You can disable the default controller temporarily by pressing L2+B on the remote controller once the robot stands up, or permanently (**recommended**) by renaming the controller executable on the robot computer with IP `192.168.123.161`.
    * After disabling the default controller, the robot should **not** stand up and should stay in motor damping mode.



2. **Setup correct permissions for non-sudo user**

   Since the Unitree SDK requires memory locking and high process priority, root priority with `sudo` is usually required to execute commands. To run the SDK without `sudo`, write the following to `/etc/security/limits.d/90-unitree.conf`:

   ```bash
   <username> soft memlock unlimited
   <username> hard memlock unlimited
   <username> soft nice eip
   <username> hard nice eip
   ```

   Log out and log back in for the above changes to take effect.

2. **Connect to the real robot**

   Connect from computer to the real robot using an Ethernet cable, and set the computer's IP address to be `192.168.123.24` (or anything in the `192.168.123.X` range that does not collide with the robot's existing IPs). Make sure you can ping/SSH into the robot's computer (by default it is `unitree@192.168.123.12`).

3. **Test connection**

   Start up the robot and make sure the robot is in joint-damping mode. Then, run the following:
   ```bash
   python -m src.robots.go1_robot_exercise_example --use_real_robot=True --use_gpu=False --num_envs=1
   ```

   The robot should be moving its body up and down following a pre-set trajectory. Terminate the script at any time to bring the robot back to joint-damping position.

## Code Structure

### Simulation

The simulation infrastructure is mostly a lightweight wrapper around `IsaacGym` that supports parallel simulation of the robot instances:
* `src/robots/robot.py` contains general robot API.
* `src/robots/go1.py` contains Go1-specific configurations.
* `src/robots/motors.py` contains motor configurations.

### Real Robot Interface

The real robot infrastructure is mostly implemented in `robots/go1_robot.py`, which invokes the C++ interface via pybind to communicate with Unitree SDKs. In addition:

* `src/robots/go1_robot_state_estimator.py` provides a simple KF-based implementation to estimate the robot's speed.

### Centroidal QP Controller

The Centroidal QP Controller is implemented in `src/controllers`:
* `src/controllers/phase_gait_generator.py` implements the the gait modulation for each leg.
* `src/controllers/qp_torque_optimizer.py` implements the torque controller for stance legs.
* `src/controllers/raibert_swing_leg_controller` implements the position controller for swing legs.

### Environments

The environment is implemented in `src/envs/jump_env.py`, where the configs can be found at `src/envs/configs`.

## Acknowledgments

This repository is inspired by, and refactored from, the [legged_gym](https://github.com/leggedrobotics/legged_gym) repository. In addition, the PPO implementation is modified from [rsl_rl](https://github.com/leggedrobotics/rsl_rl). We thank the authors of these repos for their efforts.

