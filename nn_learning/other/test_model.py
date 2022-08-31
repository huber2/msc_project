from os.path import dirname, abspath
import numpy as np
from scipy.spatial.transform import Rotation
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.errors import IKError
import warnings
import torch
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from mlp import MLP


def do_action(arm, action, ref, counter_error):
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
    try:
        q_next = arm.solve_ik_via_jacobian(next_pos, quaternion=next_quat, relative_to=ref)
    except IKError as e:
        counter_error += 1
        q_next = q


    v = (q_next - q) / env.get_simulation_timestep()
    arm.set_joint_target_velocities(v)

    env.step()
    return counter_error

    
def reset_scene(init_obj, init_robot, all_objects, arm, ref):    
    for i_obj, obj in enumerate(all_objects):
        obj.set_position(init_obj[i_obj, :3], relative_to=ref)
        obj.set_orientation(init_obj[i_obj, 3:], relative_to=ref)
    arm.set_joint_positions(init_robot, disable_dynamics=True)
    print("init joints", init_robot*180/np.pi)
    print("current joints", np.array(arm.get_joint_positions())*180/np.pi)

    
def normalize_input(x):
    return x

def denormalize_output(x, mean, std):
    return x * std + mean
    

def test_model(model, env, camera, objects, arm, ref, init_config_loader, mean, std):
    for demo, (init_obj, init_robot) in enumerate(init_config_loader):
        print(f"Demo {demo}")
        env.stop()
        env.start()
        reset_scene(init_obj, init_robot, objects, arm, ref)
        counter_error = 0
        for step in range(300):
            print(f"step #{step}")
            img_inp = torch.tensor(camera.capture_rgb().flatten(), dtype=torch.float32)[None]
            with torch.no_grad():
                act_outp = denormalize_output(model.forward(img_inp).detach().numpy().flatten(), mean, std)
            counter_error = do_action(arm, act_outp, ref, counter_error)
            if counter_error >= 10:
                warnings.warn('10x IKError', Warning)
                break
            x = input()
            if x=='b':
                break
            if x=='q':
                exit()
    env.stop()
    env.shutdown()

        


DIR_PATH = dirname(abspath(__file__)) + '/../'
SCENE_FILE = DIR_PATH + 'coppelia_scenes/Franka_bc.ttt'

model = MLP(layers_dim=(16*16*3, 128, 7))
model.load_state_dict(torch.load(DIR_PATH + 'data/model_1_demo.pth'))
model.eval()

demo_data_pth = DIR_PATH + 'data/demo_reach_object_22_07_08_18_12_37.npz'
demo_data = np.load(demo_data_pth, allow_pickle=True)
init_config_loader = zip(demo_data['init_obj_poses'], demo_data['init_robot_joints'])

mean =  np.array([ 6.76969999e-04, -8.60743590e-04, -1.71629605e-03, -1.13504647e-04,
        -7.67987723e-04, -6.42218756e-03, -9.99953806e-01])

std = np.array([4.36391173e-04, 4.05277628e-04, 1.06076294e-03, 2.58872957e-03,
        2.60789325e-03, 6.08558791e-03, 3.68366965e-05])

env = PyRep()
env.launch(SCENE_FILE, headless=False, responsive_ui=True)
arm = Panda()
arm.set_control_loop_enabled(False)
ref = arm.get_object('Reference')
camera = VisionSensor('Panda_camera')
object_names = ['blue_target', 'red_cube', 'red_cube_big', 'lime_cylinder', 'orange_sphere']

objects = [Shape(obj) for obj in object_names]

test_model(model, env, camera, objects, arm, ref, init_config_loader, mean, std)
env.shutdown()







