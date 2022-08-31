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

### build_demo_data
The `build_demo_data` directory contains the code to create the demonstration datasets of trajectories (without data augmentation):
- `create_datasets_no_ori.py` calls the `collect_demos_no_ori.py` to create demonstration trajectories **without orientation** taken into account.
- `create_datasets_ori.py` calls the `collect_demos_ori.py` to create demonstration trajectories **with orientation** taken into account.

### experiments
The `experiments` directory contains the code to learn a control policy (modeled with a neural network) from the demonstrations and test:

### results
This is were we plot the results of the experiments

### data
Please see the OneDrive link. The `data`directory is empty because of storage limit. This is were we store the demonstration data, the pytorch models of the control policy and the raw data of the results.
