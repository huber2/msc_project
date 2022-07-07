from os.path import dirname, abspath
import numpy as np
from scipy.spatial.transform import Rotation
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
import torch
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from mlp import MLP


def do_action(arm, action, ref):
    current_pos = arm.get_tip().get_position(relative_to=ref)
    current_quat = arm.get_tip().get_quaternion(relative_to=ref)

    delta_pos = action[:3]
    delta_quat = action[3:]

    next_pos = current_pos + delta_pos
    next_quat = (Rotation.from_quat(delta_quat) * Rotation.from_quat(current_quat)).as_quat()

    #print('curr_quat', list(current_quat.round(5)))
    #print('next_quat', list(next_quat.round(5)))
    #print('delt_quat', list(delta_quat.round(5)))
    #print('delta_pos', list(delta_pos.round(5)))

    #print('diatance', np.linalg.norm(delta_pos))
    #print('rotation_mag', Rotation.from_quat(delta_quat).magnitude())

    q = np.array(arm.get_joint_positions())
    q_next = arm.solve_ik_via_jacobian(next_pos, quaternion=next_quat, relative_to=ref)

    v = (q_next - q) / env.get_simulation_timestep()
    arm.set_joint_target_velocities(v)

    env.step()

    
def reset_scene(init_obj, init_robot, all_objects, arm, ref):    
    for i_obj, obj in enumerate(all_objects):
        obj.set_position(init_obj[i_obj, :3], relative_to=ref)
        obj.set_orientation(init_obj[i_obj, 3:], relative_to=ref)
    arm.set_joint_positions(init_robot, disable_dynamics=True)

    
def normalize_input(x):
    return x

def denormalize_output(x, mean, std):
    return x * std + mean
    

def test_model(model, env, camera, objects, arm, ref, init_config_loader, mean, std):
    arm.set_control_loop_enabled(False)
    for demo, (init_obj, init_robot) in enumerate(init_config_loader):
        print(f"Demo {demo}")
        env.start()
        reset_scene(init_obj, init_robot, objects, arm, ref)
        for step in range(100):
            print(f"step #{step}")
            img_inp = torch.tensor(normalize_input(camera.capture_rgb()), dtype=torch.float32)[None].permute(0, 3, 1, 2)
            with torch.no_grad():
                act_outp = denormalize_output(model.forward(img_inp).detach().numpy().flatten(), mean, std)
            do_action(arm, act_outp, ref)
            x = input()
            if x=='b':
                break
            if x=='q':
                exit()
        env.stop()


DIR_PATH = dirname(abspath(__file__)) + '/../'
SCENE_FILE = DIR_PATH + 'coppelia_scenes/Franka_bc.ttt'

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False, num_classes=7)
model.load_state_dict(torch.load(DIR_PATH + 'data/model_resnet18_20epochs_new_dataset7jul.pth'))
model.eval()

demo_data_pth = DIR_PATH + 'data/demo_reach_object_22_07_07_10_51_44.npz'
demo_data = np.load(demo_data_pth, allow_pickle=True)
init_config_loader = zip(demo_data['init_obj_poses'][:35], demo_data['init_robot_joints'][:35])

mean = np.array([ 1.43829243e-04, -1.25095708e-04, -3.39056340e-03, -9.48246625e-05,
         2.33874860e-04, -3.67510995e-04, -9.99957146e-01])
std = np.array([1.30591054e-03, 2.48425228e-03, 1.60129421e-03, 4.28247242e-03,
        4.15962985e-03, 7.06149528e-03, 3.72335535e-05])

env = PyRep()
env.launch(SCENE_FILE, headless=False, responsive_ui=True)
arm = Panda()
ref = arm.get_object('Reference')
camera = VisionSensor('Panda_camera')
object_names = ['blue_target', 'red_cube', 'red_cube_big', 'lime_cylinder', 'orange_sphere']

objects = [Shape(obj) for obj in object_names]

test_model(model, env, camera, objects, arm, ref, init_config_loader, mean, std)

env.shutdown()








