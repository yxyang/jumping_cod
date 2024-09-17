Simulation and Training Setup
=============================

System Requirement
------------------
The training environment requires an X86-architectured computer and a Nvidia GPU with at least 12GB of RAM (Recommended: RTX 3090, RTX 4090 or better.)

The code has been tested on Ubuntu 20.04 and Ubuntu 22.04.


Installing Related Packages
---------------------------
1. Create a new virtual environment under Python 3.6, 3.7, 3.8 (3.8 recommended).

2. Install dependencies:

   .. code-block:: bash

      pip install -r requirements.txt

   Note that the ``numpy`` version must be no later than ``1.19.5`` (already specified in ``requirements.txt``) to avoid conflict with the Isaac Gym utility files.

3. Download and install IsaacGym Preview 4:

   - Download IsaacGym from `https://developer.nvidia.com/isaac-gym <https://developer.nvidia.com/isaac-gym>`_. Extract the downloaded file to the root folder.

   - Install the downloaded IsaacGym.

   .. code-block:: bash

         cd isaacgym/python && pip install -e .

   - Try running the example:

     .. code-block:: bash

         cd examples && python 1080_balls_of_solitude.py

     The code is set to run on CPU so don't worry if you see an error about the GPU not being utilized.

4. Install `rsl_rl` (adapted PPO implementation):

   .. code-block:: bash

      cd rsl_rl && pip install -e .

5. Lastly, build and install the interface to Unitree's Go1 SDK. The Unitree `repo <https://github.com/unitreerobotics/unitree_legged_sdk>`_ has been releasing new SDK versions. For convenience, we have included the version that we used in ``third_party/unitree_legged_sdk``.

   - First, make sure the required packages are installed, following Unitree's `guide <https://github.com/unitreerobotics/unitree_legged_sdk>`_. Most notably, please make sure to install ``Boost`` and ``LCM``:

     .. code-block:: bash

        sudo apt install libboost-all-dev liblcm-dev

   - Then, go to ``third_party/go1_sdk`` and create a build folder:

     .. code-block:: bash

        cd third_party/go1_sdk
        mkdir build && cd build

   - Now, build the libraries and move them to the main directory by running:

     .. code-block:: bash

        cmake ..
        make
        mv go1_interface* ../../..

Running the Demo Policy
-------------------

We provide a demo policy of stair jumping. To run the policy with ground-truth heightmap, run:

.. code-block:: bash

  python -m src.agents.ppo.eval --logdir=data/demo_policy/stair_rl/model_8000.pt --num_envs=1 --show_gui=True --use_real_robot=False --save_traj=False


To run the policy with estimated heightmap, run:

.. code-block:: bash

   python -m src.agents.heightmap_prediction.eval --logdir=data/demo_policy/stair_distill/model_29.pt --save_traj=False --num_envs=1

