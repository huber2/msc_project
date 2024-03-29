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
SCENE_FILE = DIR_PATH + 'coppelia_scenes/Franka_2cuboids_32camera.ttt'


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


def get_test_acc(n_tests, model, env, camera, arm, objects, target_dummy, ref, max_steps, distance_tolerance, maintain_target_duration):
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
        if is_reached:
            counter_reached +=1
        print(f'Test {i+1}/{n_tests} ; Total reached: {counter_reached}/{i+1} ; running accuracy {counter_reached/(i+1)}')

    return counter_reached / n_tests

def format_color(color256):
    """Convert from [0-255] format to [0-1] format"""
    return [c/255 for c in color256]


def main():
    model_name = "nn32_FF0000_aug_model"
    model_path = DIR_PATH + "data/" + model_name + ".pth"
    n_tests = 2
    max_steps = 300
    distance_tolerance = 0.05
    maintain_target_duration = 10
    target_color = [255, 0, 0]
    distractor_colors = [
                [0, 0, 0], 
                [255, 255, 255], 
                [255, 255, 0], 
                [255, 0, 255], 
                [128, 0, 0], 
                [192, 0, 0], 
                [230, 0, 0], 
                [255, 0, 0], 
                [255, 128, 128], 
                [256, 26, 26]
                ]
    random_seed = 456

    model = ConvNet(n_classes=3)
    model.load_state_dict(torch.load(model_path))
    print('MODEL PATH', model_path)
    print('MODEL INFO:', model, sep='\n')
    model.eval()
    
    env = PyRep()
    env.launch(SCENE_FILE, headless=False, responsive_ui=False)
    arm = Panda()
    ref = arm.get_object('Reference')
    camera = VisionSensor('Panda_camera')
    target_dummy = Dummy('target_dummy')
    object_names = ['cuboid0', 'cuboid1']
    objects = [Shape(obj) for obj in object_names]
    info = {
        "target_color": target_color,
        "distractor_colors": distractor_colors,
        "acc_list": [],
    }

    print('Generating initial random object poses...')
    #init_configs = get_scene_random_initial_positions(n_tests, len(objects))
    print('Running tests...')
    print(f'A tests passes if the robot tip reaches the target position witin {distance_tolerance}m and stays there for {maintain_target_duration} time steps.')
    print(f'Otherwise, if this is not the case after {max_steps} time steps, the test fails.')

    for distractor_color in distractor_colors:
        np.random.seed(random_seed)
        env.stop()
        objects[0].set_color(format_color(target_color))
        objects[1].set_color(format_color(distractor_color))
        acc = get_test_acc(n_tests, model, env, camera, arm, objects, target_dummy, ref, max_steps, distance_tolerance, maintain_target_duration)
        info["acc_list"].append(acc)
        print(f'All test done for distractor color:{distractor_color}! Final accuracy:', acc)
        print(info)
    env.shutdown()

    with open(f"{DIR_PATH}data/test_{model_name}.json", 'w') as f:
        json.dump(info, f)

if __name__ == "__main__":
    main()