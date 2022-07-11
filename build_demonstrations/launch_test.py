from os.path import dirname, abspath
import numpy as np
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
from simple_test import test_model
from nn_learning import MLP


DIR_PATH = dirname(abspath(__file__)) + '/../'
SCENE_FILE = DIR_PATH + 'coppelia_scenes/Franka_bc.ttt'

model = MLP(layers_dim=(16*16*3, 16, 3))
model.load_state_dict(torch.load(DIR_PATH + 'data/model_simple.pth'))
model.eval()

normalization_info = np.load('data/model_simple_normalization_info.npz')
mean = normalization_info['mean'] 
std = normalization_info['std'] 

env = PyRep()
env.launch(SCENE_FILE, headless=False, responsive_ui=True)
arm = Panda()
ref = arm.get_object('Reference')
camera = VisionSensor('Panda_camera')
test_model(model, env, camera, objects, arm, ref, init_config_loader, mean, std, max_steps)
env.shutdown()