from os.path import dirname, abspath
import numpy as np
import torch
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from simple_test import test_model
from model_simple_conv import ConvNet
from mlp import MLP


DIR_PATH = dirname(abspath(__file__)) + '/../'
SCENE_FILE = DIR_PATH + 'coppelia_scenes/Franka_big_distractors.ttt'
model_path = DIR_PATH + 'data/model_conv_simple_100ep_50_16x16_mask5x5_v_xyz.pth'

model = ConvNet(n_classes=3)
model.load_state_dict(torch.load(model_path))
#model = MLP(layers_dim=(16*16*3, 64, 3))
#model.load_state_dict(torch.load(DIR_PATH + 'data/model_mlp_simple_2sides.pth'))
print(model_path)
print('MODEL INFO:', model, sep='\n')
model.eval()

# normalization_info = np.load('data/model_simple_normalization_info.npz')
# mean = normalization_info['mean'] 
# std = normalization_info['std']

def format_input(x):
    return torch.tensor(x, dtype=torch.float32)[None].permute(0, 3, 1, 2)


def format_output(x):
    return x.detach().numpy().flatten() / 200

max_steps = 200

env = PyRep()
env.launch(SCENE_FILE, headless=False, responsive_ui=True)
arm = Panda()
ref = arm.get_object('Reference')
camera = VisionSensor('Panda_camera')
target_obj = Shape('blue_target')
target_dummy = Dummy('target_dummy')
target_pos = np.array(target_obj.get_position(relative_to=ref))
target_pos1 = np.array([target_pos[0], 0.25, target_pos[2]])
target_pos2 = np.array([target_pos[0], -0.25, target_pos[2]])
target_pos3 = np.array([target_pos[0]-0.2, +0.07, target_pos[2]])
target_pos4 = np.array([target_pos[0]+0.1, 0, target_pos[2]])
target_pos5 = np.array([target_pos[0]+0.06, -0.3, target_pos[2]])
target_pos6 = np.array([target_pos[0]-0.2, +0.35, target_pos[2]])

targets = (target_pos1, target_pos2, target_pos3, target_pos4, target_pos5, target_pos6)

for i, tgt in enumerate(targets):
    print(f'Test {i+1}/{len(targets)}')
    target_obj.set_position(tgt, relative_to=ref)
    test_model(model, env, camera, arm, ref, max_steps, format_input, format_output)

env.shutdown()