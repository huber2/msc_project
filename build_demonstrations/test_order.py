from os.path import dirname, abspath
import numpy as np
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.dummy import Dummy


DIR_PATH = dirname(abspath(__file__)) + '/../'
SCENE_FILE = DIR_PATH + 'coppelia_scenes/Franka_blue_cuboid_32camera.ttt'


env = PyRep()
env.launch(SCENE_FILE, headless=False, responsive_ui=False)
arm = Panda()
arm.set_control_loop_enabled(False)
ref = arm.get_object('Reference')
camera = VisionSensor('Panda_camera')
target_object = Shape('blue_target')
target_dummy = Dummy('target_dummy')

imgs = []


env.start()

#1
imgs.append(camera.capture_rgb())

target_object.set_color([1, 0, 0])

#2
imgs.append(camera.capture_rgb())

env.step()

#3
imgs.append(camera.capture_rgb())

target_object.set_color([1, 1, 0])

#4
imgs.append(camera.capture_rgb())

env.step()

#5
imgs.append(camera.capture_rgb())

arm.set_joint_target_velocities([1]*7)

#6
imgs.append(camera.capture_rgb())

env.step()

#7
imgs.append(camera.capture_rgb())

env.step()

#8
imgs.append(camera.capture_rgb())


env.stop()
env.shutdown()

imgs_array = np.array(np.stack(imgs) * 255, dtype=np.uint8)
np.savez(DIR_PATH + 'data/test_order_images', x=imgs_array)