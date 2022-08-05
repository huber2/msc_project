from os.path import dirname, abspath
from datetime import datetime
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.dummy import Dummy
from simple_reach_with_ori import collect_and_save_demos


DIR_PATH = dirname(abspath(__file__)) + '/../'
SCENE_FILE = DIR_PATH + 'coppelia_scenes/Franka_blue_cuboid_32camera.ttt'

timestamp = datetime.now().strftime('%y_%m_%d_%_H_%M_%S')
save_location = DIR_PATH + 'data/demo_reach_object_' + timestamp


demo_params = {
    'n_demos': 1500,
    'max_steps': 500,
    'max_speed_linear': 0.1,
    'max_speed_angular': 0.2,
    'precision_linear': 0.005,
    'precision_angular': 0.01,
    'maintain': 30,
    'save_demo_location': save_location,
    'headless': True,
}


env = PyRep()
env.launch(SCENE_FILE, headless=demo_params['headless'], responsive_ui=False)

arm = Panda()
ref = arm.get_object('Reference')
camera = VisionSensor('Panda_camera')
target_object = Shape('blue_target')
target_object.set_color([0, 0, 0])
target_dummy = Dummy('target_dummy')

collect_and_save_demos(env, arm, camera, target_object, target_dummy, ref, **demo_params)
