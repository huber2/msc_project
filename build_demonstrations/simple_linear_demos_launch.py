from os.path import dirname, abspath
from datetime import datetime
import numpy as np
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.dummy import Dummy
from simple import collect_and_save_demos


DIR_PATH = dirname(abspath(__file__)) + '/../'
SCENE_FILE = DIR_PATH + 'coppelia_scenes/Franka_bc.ttt'

timestamp = datetime.now().strftime('%y_%m_%d_%_H_%M_%S')
save_location = DIR_PATH + 'data/demo_reach_object_' + timestamp


demo_params = {
    'n_demos': 1,
    'max_steps': 500,
    'max_speed_linear': 0.05,
    'precision_linear': 0.005,
    'maintain': 20,
    'save_demo_location': save_location,
}

scene_object_names = [
    'blue_target',
    'red_cube',
    'red_cube_big',
    'lime_cylinder',
    'orange_sphere',
]

env = PyRep()
env.launch(SCENE_FILE, headless=False, responsive_ui=False)

arm = Panda()
ref = arm.get_object('Reference')
camera = VisionSensor('Panda_camera')
objects = [Shape(obj) for obj in scene_object_names]
target_object = objects[0]
distractors = objects[1:]
target_dummy = Dummy('target_dummy')

collect_and_save_demos(env, arm, camera, target_object, target_dummy, distractors, ref, **demo_params)
env.shutdown()

