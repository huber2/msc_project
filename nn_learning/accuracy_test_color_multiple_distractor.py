from os.path import dirname, abspath
import json
import numpy as np
import torch
import warnings
from pyrep import PyRep
from pyrep.errors import IKError
from pyrep.robots.arms.panda import Panda
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from model_very_simple_conv_32 import ConvNet
from pyrep.errors import IKError
import warnings

DIR_PATH = dirname(abspath(__file__)) + '/../'
SCENE_FILE = DIR_PATH + 'coppelia_scenes/Franka_distractors_same_shape_many_colors_32.ttt'


def format_input(x):
    return torch.tensor(x, dtype=torch.float32)[None].permute(0, 3, 1, 2)


def format_output(x):
    return x.detach().numpy().flatten() / 10

def set_controls(env, arm, action, ref, counter_error, init_quat):
    current_pos = arm.get_tip().get_position(relative_to=ref)
    next_pos = current_pos + action * env.get_simulation_timestep()
    next_quat = init_quat
    q = np.array(arm.get_joint_positions())
    try:
        q_next = arm.solve_ik_via_jacobian(next_pos, quaternion=next_quat, relative_to=ref)
    except IKError as e:
        print('IKError: setting joint_target_v to 0')
        arm.set_joint_target_velocities([0] * arm.get_joint_count())
        counter_error += 1
        return counter_error
    v = (q_next - q) / env.get_simulation_timestep()
    arm.set_joint_target_velocities(v)
    return counter_error


def reached_condition(arm, target_dummy, distance_tolerance):
    return arm.get_tip().check_distance(target_dummy) < distance_tolerance



def test_model(model, env, camera, arm, target_dummy, ref, max_steps, distance_tolerance, maintain_target_duration, objects):
    counter_error = 0
    counter_target_reached = 0
    init_quat = arm.get_tip().get_quaternion(relative_to=ref)
    for step in range(max_steps):
        #print(f"step #{step}/{max_steps}", end=" ")
        env.step() # images are caputred at the end of the last step() call
        img_input = camera.capture_rgb()
        with torch.no_grad():
            x = format_input(img_input)
            y = model.forward(x)
            act_output = format_output(y)
        counter_error = set_controls(env, arm, act_output, ref, counter_error, init_quat)

        if counter_error >= 10:
            env.stop()
            warnings.warn('10x IKError', Warning)
            break

        if reached_condition(arm, target_dummy, distance_tolerance):
            counter_target_reached += 1

        if counter_target_reached >= maintain_target_duration:
            env.stop()
            return True
        """
        x = input()
        if x=='b':
            break
        if x=='q':
            exit()
        """
    return False


def multiple_tests(n_tests, model, env, camera, arm, objects, target_dummy, ref, max_steps, distance_tolerance, maintain_target_duration):
    is_reached_list = []
    obj_bounding_box = np.array([0.55, -0.25, 0.8, 0.8, 0.25, 0.8])
    obj_bounding_euler_angles = np.array([180, 0, 90, 180, 0, 270]) * np.pi/180
    obj_min = np.concatenate([obj_bounding_box[:3], obj_bounding_euler_angles[:3]]).reshape(1, 6)
    obj_max = np.concatenate([obj_bounding_box[3:], obj_bounding_euler_angles[3:]]).reshape(1, 6)
    counter_reached = 0
    arm.set_control_loop_enabled(False)
    for i in range(n_tests):
        env.stop()
        
        for i_obj, obj in enumerate(objects):
            is_collision = True
            while is_collision:
                is_collision = False
                init_poses = np.random.uniform(obj_min, obj_max, (1, 6)).flatten()
                obj.set_position(init_poses[:3], relative_to=None)
                obj.set_orientation(init_poses[3:])
                for j_obj in range(i_obj):
                    if objects[i_obj].check_collision(objects[j_obj]):
                        is_collision = True
                        break

        env.start()
        is_reached = test_model(model, env, camera, arm, target_dummy, ref, max_steps, distance_tolerance, maintain_target_duration, objects)
        env.stop()
        is_reached_list.append(is_reached)
        print(f'Test {i+1}/{n_tests} ; Total reached: {sum(is_reached_list)}/{len(is_reached_list)} ; running accuracy {sum(is_reached_list)/len(is_reached_list)}')

    return is_reached_list

def format_color(color256):
    """Convert from [0-255] format to [0-1] format"""
    return [c/255 for c in color256]


def main():
    model_name = "nn32_0000FF_aug_model"
    model_path = DIR_PATH + "data/" + model_name + ".pth"
    n_tests = 5
    max_steps = 300
    distance_tolerance = 0.05
    maintain_target_duration = 10

    a = np.array([0,0,1])
    b = np.array([1,1,0])
    range_col = np.round(np.arange(0.1, 1.01, 0.1) * 255)
    aa = range_col.reshape((-1, 1)) * a.reshape((1, -1))
    bb = range_col.reshape((-1, 1)) * b.reshape((1, -1)) + a*255
    target_colors = np.array(np.concatenate([aa, bb])[:-1], dtype=np.uint8)
    random_seed = 456

    model = ConvNet(n_classes=3)
    model.load_state_dict(torch.load(model_path))
    print('MODEL PATH', model_path)
    print('MODEL INFO:', model, sep='\n')
    model.eval()
    
    env = PyRep()
    env.launch(SCENE_FILE, headless=True, responsive_ui=False)
    arm = Panda()
    ref = arm.get_object('Reference')
    camera = VisionSensor('Panda_camera')
    target_dummy = Dummy('target_dummy')
    object_names = [
        'blue_cuboid', 
        'red_cuboid', 
        'green_cuboid', 
        'white_cuboid', 
        'black_cuboid',
        'yellow_cuboid',
        'purple_cuboid',
        'turquoise_cuboid',
        ]
    objects = [Shape(obj) for obj in object_names]
    sucess_results = []

    print('Generating initial random object poses...')
    #init_configs = get_scene_random_initial_positions(n_tests, len(objects))
    print('Running tests...')
    print(f'A tests passes if the robot tip reaches the target position witin {distance_tolerance}m and stays there for {maintain_target_duration} time steps.')
    print(f'Otherwise, if this is not the case after {max_steps} time steps, the test fails.')

    for target_color in target_colors:
        np.random.seed(random_seed)
        env.stop()
        objects[0].set_color(format_color(target_color))
        is_reached_list = multiple_tests(n_tests, model, env, camera, arm, objects, target_dummy, ref, max_steps, distance_tolerance, maintain_target_duration)
        sucess_results.append(is_reached_list)
        print(f'All test done! Final accuracy:')
        print(sum(is_reached_list)/len(is_reached_list))
    env.shutdown()

    sucess_results = np.array(sucess_results)
    results = {
        "target_colors": target_colors,
        "sucess_results": sucess_results,
    }
    np.savez_compressed(f"{DIR_PATH}data/test_many_distractors_{model_name}", **results)

if __name__ == "__main__":
    main()