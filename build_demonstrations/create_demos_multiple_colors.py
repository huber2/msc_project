from os.path import dirname, abspath
from datetime import datetime
import numpy as np
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.dummy import Dummy
from create_demos import collect_and_save_demos

def rgb256_to_hex(color256):
    r, g, b = color256
    return f"{r:02X}{g:02X}{b:02X}"

def rgb256_to_rgb1(color256):
    return [i/255 for i in color256]

DIR_PATH = dirname(abspath(__file__)) + '/../'
SCENE_FILE = DIR_PATH + 'coppelia_scenes/Franka_blue_cuboid_32camera.ttt'

timestamp = datetime.now().strftime('%y_%m_%d_%_H_%M_%S')
#save_location = DIR_PATH + 'data/demo_reach_object_' + timestamp
save_location = DIR_PATH + 'data/demo_reach_white32x32'

demo_params = {
    'n_demos': 3,
    'tgt_bounding_box': np.array([0.35, -0.4, 0.03, 0.85, 0.4, 0.03]),
    'tgt_bounding_angles': np.array([180, 0, -180, 180, 0, 180]) * np.pi/180,
    'max_steps': 500,
    'max_speed_linear': 0.1,
    'precision_linear': 0.005,
    'maintain': 30,
    'seed': 123,
    'headless': False,
}

colors = (
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
)

for color256 in colors:
    demo_params['save_location'] = DIR_PATH + 'data/demo_reach_cuboid_' + rgb256_to_hex(color256) + '_camera32'
    env = PyRep()
    env.launch(SCENE_FILE, headless=demo_params['headless'], responsive_ui=False)
    arm = Panda()
    ref = arm.get_object('Reference')
    camera = VisionSensor('Panda_camera')
    target_object = Shape('blue_target')
    target_object.set_color(rgb256_to_rgb1(color256))
    target_dummy = Dummy('target_dummy')
    collect_and_save_demos(env, arm, camera, target_object, target_dummy, ref, **demo_params)
