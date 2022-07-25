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
from pyrep.errors import IKError
import warnings


DIR_PATH = dirname(abspath(__file__)) + '/../'
SCENE_FILE = DIR_PATH + 'coppelia_scenes/Franka_big_distractors.ttt'
model_path = DIR_PATH + 'data/model_conv_simple_100ep_50_16x16_mask5x5_v_xyz.pth'
n_tests = 100
max_steps = 200

model = ConvNet(n_classes=3)
model.load_state_dict(torch.load(model_path))
#model = MLP(layers_dim=(16*16*3, 64, 3))
#model.load_state_dict(torch.load(DIR_PATH + 'data/model_mlp_simple_2sides.pth'))
print('MODEL PATH', model_path)
print('MODEL INFO:', model, sep='\n')
model.eval()

# normalization_info = np.load('data/model_simple_normalization_info.npz')
# mean = normalization_info['mean'] 
# std = normalization_info['std']

def format_input(x):
    return torch.tensor(x, dtype=torch.float32)[None].permute(0, 3, 1, 2)


def format_output(x):
    return x.detach().numpy().flatten() / 200


env = PyRep()
env.launch(SCENE_FILE, headless=False, responsive_ui=True)
arm = Panda()
ref = arm.get_object('Reference')
camera = VisionSensor('Panda_camera')
target_dummy = Dummy('target_dummy')

object_names = ['blue_target', 'red_cube', 'red_cube_big', 'lime_cylinder', 'orange_sphere']
objects = [Shape(obj) for obj in object_names]


def get_scene_random_initial_positions(n_tests, n_objects):
    obj_bounding_box = [0.5, -0.2, 0.8, 0.8, 0.2, 0.8]
    obj_bounding_euler_angles = np.array([180, 0, -180, 180, 0, 180]) * np.pi/180
    obj_min = np.concatenate([obj_bounding_box[:3], obj_bounding_euler_angles[:3]]).reshape(1, 6)
    obj_max = np.concatenate([obj_bounding_box[3:], obj_bounding_euler_angles[3:]]).reshape(1, 6)
    all_obj_init_poses = np.random.uniform(obj_min, obj_max, (n_tests, n_objects, 6))
    return all_obj_init_poses

def set_controls(env, arm, action, ref, counter_error, init_quat):
    current_pos = arm.get_tip().get_position(relative_to=ref)
    next_pos = current_pos + action
    next_quat = init_quat
    q = np.array(arm.get_joint_positions())
    try:
        q_next = arm.solve_ik_via_jacobian(next_pos, quaternion=next_quat, relative_to=ref)
    except IKError as e:
        arm.set_joint_target_velocities([0] * arm.get_joint_count())
        counter_error += 1
        return counter_error
    v = (q_next - q) / env.get_simulation_timestep()
    arm.set_joint_target_velocities(v)
    return counter_error
    

def test_model(model, env, camera, arm, target_dummy, ref, max_steps, format_input, format_output):
    init_quat = arm.get_tip().get_quaternion(relative_to=ref)
    counter_error = 0
    counter_target_reached = 0
    for step in range(max_steps):
        #print(f"step #{step}/{max_steps}", end=" ")
        img_input = camera.capture_rgb()
        env.step() # Important to get the image
        with torch.no_grad():
            x = format_input(img_input)
            y = model.forward(x)
            act_output = format_output(y)
        counter_error = set_controls(env, arm, act_output, ref, counter_error, init_quat)

        if counter_error >= 10:
            warnings.warn('10x IKError', Warning)
            break

        if arm.get_tip().check_distance(target_dummy) < 0.05:
            counter_target_reached += 1

        if counter_target_reached >= 10:
            return True
        """
        x = input()
        if x=='b':
            break
        if x=='q':
            exit()
        """
    return False


def get_test_acc(init_configs, model, env, camera, arm, objects, target_dummy, ref, max_steps, format_input, format_output):
    counter_reached = 0
    arm.set_control_loop_enabled(False)
    for i, init_poses in enumerate(init_configs):
        env.stop()
        env.start()
        for idx_obj, obj in enumerate(objects):
            obj.set_position(init_poses[idx_obj, :3], relative_to=None)
            obj.set_orientation(init_poses[idx_obj, 3:])
        
        is_reached = test_model(model, env, camera, arm, target_dummy, ref, max_steps, format_input, format_output)
        if is_reached:
            counter_reached +=1
        print(f'Test {i+1}/{len(init_poses)} ; Total reached: {counter_reached}/{i+1} ; running accuracy {counter_reached/(i+1)}')
    env.stop()
    env.shutdown()
    return counter_reached / len(init_poses)



init_configs = get_scene_random_initial_positions(n_tests, len(objects))
acc = get_test_acc(init_configs, model, env, camera, arm, objects, target_dummy, ref, max_steps, format_input, format_output)

print('All test done! Final accuracy:', acc)