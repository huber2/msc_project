# MSc Project: Data augmentation for robust Behavioural Cloning

This is the code used for the project. It contains the code used for the experiments.

## Dependencies


- CoppeliaSim Edu version 4.1.0 (https://www.coppeliarobotics.com/previousVersions)
- Python 3.8 (other versions > 3.6 should also work)
- PyRep (https://github.com/stepjam/PyRep). This requires a Linux OS. We used Ubuntu 20.04, the CoppeliaSim version should match the OS.

Python packages:
- torch
- numpy
- scipy
- matplotlib
- Pillow

## Organisation of files

### coppelia_scenes
The `coppelia_scenes` directory contains the scenes of the simulation environment in CoppeliaSim which are used for collecting demonstrations and testing the robot.

### build_demonstrations
The `build_demonstrations` directory contains the code to create the demonstration datasets of trajectories (without data augmentation).

- [`create_demos_multiple_colors.py`](./build_demonstrations/create_demos_multiple_colors.py) calls the [`create_demos.py`](./build_demonstrations/create_demos.py) to create demonstration trajectories **without orientation** taken into account. The high-level parameters for the training are specified in the `create_demos_multiple_colors.py` file.

- [`create_demos_multiple_colors_ori.py`](./build_demonstrations/create_demos_multiple_colors_ori.py) calls the [`create_demos_ori.py`](./build_demonstrations/create_demos_ori.py) file to create demonstration trajectories **with orientation** taken into account.

- All the other files are just previous versions of the code and drafts.

The task for the robot consist in reaching either a red a green or a blue cuboid with the shortests path. The simulation scene used for the demonstrtions  can be found in the [`Franka_blue_cuboid_32camera.ttt`](./coppelia_scenes/Franka_blue_cuboid_32camera.ttt) file. The colour of the target is changed within the python code.

### nn_learning
The `nn_learning` directory contains the code for the experiments: training neural networks, applying data augmentation, testing the policy in CoppeliaSim.

- The model of the CNN used to learn the policy is in the [`model_very_simple_conv_32.py`](./nn_learning/model_very_simple_conv_32.py) file.

- The data augmentation and the model training is done a Jupyter notebook: [`learn_reach_different_colors_plus_augmentation.ipynb`](./nn_learning/learn_reach_different_colors_plus_augmentation.ipynb). All the policies are trained with this notebook.

- The following files are for testing the models (measuring the success rate and/or the number of steps needed to reach the target):
  - [accuracy_test_color_multiple_distractor.py](./nn_learning/accuracy_test_color_multiple_distractor.py) (also used to test with one or no distractors).
  - [accuracy_test_multiple_size_distractor.py ](./nn_learning/accuracy_test_multiple_size_distractor.py)
  - [accuracy_test_multiple_models_ORI.py ](./nn_learning/accuracy_test_multiple_models_ORI.py)
  - [automated_accuracy_test_color_1distractor.py](./nn_learning/automated_accuracy_test_color_1distractor.py)
  - [accuracy_test_cone_distractor.py](./nn_learning/accuracy_test_cone_distractor.py)
  - [accuracy_test_ori.py](./nn_learning/accuracy_test_ori.py)
  
### analysis
This is were we plot the results of the experiments.
 - In the notebook [`draw_results.ipynb`](./analysis/results/draw_results.ipynb), we plot the figures of results used in the report.

### data
Please see the OneDrive link. The `data`directory is empty because of storage limit. This is were we store the demonstration data, the pytorch models of the control policy and the raw data of the results.
