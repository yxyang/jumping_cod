Jumping Continuously over Discontinuous Terrains
=========================================

+-------------------------------------+-----------------------------------------------+
| .. image:: images/teaser.gif        | .. image:: images/teaser_stepstone.gif        |
|    :scale: 50 %                     |    :scale: 50 %                               |
|    :alt: First GIF                  |    :alt: Second GIF                           |
+-------------------------------------+-----------------------------------------------+


This repository contains the code for the paper `Agile Continuous Jumping in Discontinuous Terrains <http://yxyang.github.io/jumping_cod>`_. The name of this repo, Jumping CoD, stands for *Jumping Continuously over Discontinuities*.


+-------------------------------------+
| .. image:: images/jumping_cod.jpg   |
|    :scale: 25 %                     |
|    :alt: A Jumping Cod              |
+-------------------------------------+
|          **A Jumping Cod**          |
+-------------------------------------+



The main contents of the repo includes:

* The simulation environment to train the terrain-aware jumping controller.
* The code to deploy the trained controller to a real Unitree Go1 robot.
* Additional utilities to inspect robot logs and record data for real-to-sim study.


Contents
--------

.. toctree::
   simulation_setup
   real_robot_setup
   training_a_new_policy
   utilities

Credits
-------
The perception pipeline used in this project is heavily inspired from the `Robot Parkour Learning work <https://robot-parkour.github.io/>`_. More specifically, we design our camera mount based on their 3D design, and used a very similar perception network structure. We thank the authors, especially Zipeng Fu, for his generous support.
