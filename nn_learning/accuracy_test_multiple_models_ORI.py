from os.path import dirname, abspath
import json
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import warnings
from pyrep import PyRep
from pyrep.const import PrimitiveShape
from pyrep.errors import IKError
from pyrep.robots.arms.panda import Panda
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from model_very_simple_conv_32 import ConvNet
from pyrep.errors import IKError
import warnings

DIR_PATH = dirname(abspath(__file__)) + '/../'
SCENE_FILE = DIR_PATH + 'coppelia_scenes/Franka_blue_cuboid_32camera.ttt'
#SCENE_FILE = DIR_PATH + 'coppelia_scenes/Franka_2cuboids_32camera.ttt'


def format_input(x):
    return torch.tensor(x, dtype=torch.float32)[None].permute(0, 3, 1, 2)


def format_output(x, max_speed_linear=0.1):
    x_array = x.detach().numpy().flatten()
    return x_array * max_speed_linear


def set_controls_ori(env, arm, action, ref, counter_error):
    current_pos = arm.get_tip().get_position(relative_to=ref)
    current_quat = arm.get_tip().get_quaternion(relative_to=ref)
    current_rot = Rotation.from_quat(current_quat)

    v_linear = action[:3]
    v_angular = action[3:]
    delta_angles = v_angular * env.get_simulation_timestep()
    delta_rot = Rotation.from_euler('xyz', delta_angles, degrees=False)

    next_pos = current_pos + v_linear * env.get_simulation_timestep()
    next_rot = delta_rot * current_rot
    next_quat = next_rot.as_quat()
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



def reached_condition(arm, target_dummy, distance_tolerance, angle_tolerance):
    linear_cond = arm.get_tip().check_distance(target_dummy) < distance_tolerance

    delta_quat = arm.get_tip().get_quaternion(relative_to=target_dummy)
    delta_rot = Rotation.from_quat(delta_quat)

    rot_zplus180 = Rotation.from_quat(np.array([0, 0, 1, 0]))
    rot_zminus180 = Rotation.from_quat(np.array([0, 0, -1, 0]))

    delta_rot_plus180 = rot_zplus180 * delta_rot
    delta_rot_minus180 = rot_zminus180 * delta_rot
    
    angular_cond = False
    for delta_ori in (delta_rot, delta_rot_plus180, delta_rot_minus180):
        if delta_ori.magnitude() < angle_tolerance:
            angular_cond = True
            break

    return linear_cond and angular_cond



def test_model(model, env, camera, arm, target_dummy, ref, max_steps, distance_tolerance, angle_tolerance, maintain_target_duration):
    env.step()
    counter_error = 0
    counter_target_reached = 0
    for step in range(max_steps):
        img_input = camera.capture_rgb() # images are caputred at the end of the last step() call
        with torch.no_grad():
            x = format_input(img_input)
            y = model.forward(x)
            act_output = format_output(y)
        counter_error = set_controls_ori(env, arm, act_output, ref, counter_error)

        if counter_error >= 10:
            env.stop()
            warnings.warn('10x IKError', Warning)
            break

        env.step()

        if reached_condition(arm, target_dummy, distance_tolerance, angle_tolerance):
            counter_target_reached += 1
        else:
            counter_target_reached = 0

        if counter_target_reached >= maintain_target_duration:
            env.stop()
            return True, step+1

    return False, max_steps


def multiple_tests(n_tests, model, env, camera, arm, objects, target_dummy, ref, max_steps, distance_tolerance, angle_tolerance, maintain_target_duration):
    is_reached_list = []
    n_steps_list = []
    obj_bounding_box = np.array([0.55, -0.25, 0.8, 0.8, 0.25, 0.8])
    obj_bounding_euler_angles = np.array([180, 0, 90, 180, 0, 270]) * np.pi/180
    obj_min = np.concatenate([obj_bounding_box[:3], obj_bounding_euler_angles[:3]]).reshape(1, 6)
    obj_max = np.concatenate([obj_bounding_box[3:], obj_bounding_euler_angles[3:]]).reshape(1, 6)
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
        results = test_model(model, env, camera, arm, target_dummy, ref, max_steps, distance_tolerance, angle_tolerance, maintain_target_duration)
        env.stop()
        is_reached, n_steps = results
        is_reached_list.append(is_reached)
        n_steps_list.append(n_steps)
        print(f'Test {i+1}/{n_tests} ; Total reached: {sum(is_reached_list)}/{len(is_reached_list)} ; running accuracy {sum(is_reached_list)/len(is_reached_list)}')

    return is_reached_list, n_steps_list

def format_color(color256):
    """Convert from [0-255] format to [0-1] format"""
    return [c/255 for c in color256]


def save_results(save_loc, model_names, sucess_results, steps_needed, i):
    all_sucess_results = np.array(sucess_results)
    all_steps_needed = np.array(steps_needed)
    results = {
        "model_names": model_names[:i+1],
        "sucess_results": all_sucess_results,
        "n_steps": all_steps_needed,
    }
    np.savez_compressed(save_loc, **results)


def main():
    model_names = [
        "nn32_ori_no_aug_model_0000FF_ntraj_250",
        "nn32_ori_no_aug_model_0000FF_ntraj_200",
        "nn32_ori_no_aug_model_0000FF_ntraj_150",
        "nn32_ori_no_aug_model_0000FF_ntraj_100",
        "nn32_ori_no_aug_model_0000FF_ntraj_50",
        "nn32_ori_no_aug_model_0000FF_ntraj_10",
        ]

    max_steps = 500
    distance_tolerance = 0.02
    angle_tolerance = 0.2
    maintain_target_duration = 10
    random_seed = 456
    n_tests = 100


    env = PyRep()
    env.launch(SCENE_FILE, headless=True, responsive_ui=False)
    arm = Panda()
    ref = arm.get_object('Reference')
    camera = VisionSensor('Panda_camera')
    target_dummy = Dummy('target_dummy')
    

    target_name = "cuboid0"
    target_color = [0, 0, 255]
    dist_colors = []

    sucess_results = []
    steps_needed = []

    print('Generating initial random object poses...')
    #init_configs = get_scene_random_initial_positions(n_tests, len(objects))
    print('Running tests...')
    print(f'A tests passes if the robot tip reaches the target position witin {distance_tolerance}m and stays there for {maintain_target_duration} time steps.')
    print(f'Otherwise, if this is not the case after {max_steps} time steps, the test fails.')

    for i, model_name in enumerate(model_names):

        model = ConvNet(n_classes=6)
        model_path = DIR_PATH + "data/" + model_name + ".pth"
        model.load_state_dict(torch.load(model_path))
        model.eval()

        save_loc = f"{DIR_PATH}data/test_ori_blue_models_no_aug_no_distractor"

        print('MODEL PATH', model_path)
        print('MODEL INFO:', model, sep='\n')
        print("target color", target_color)

        np.random.seed(random_seed)

        env.stop()

        dist_size = [0.12, 0.06, 0.06]
        distrators = []
        for col in dist_colors:
            distrators.append(Shape.create(PrimitiveShape.CUBOID, size=dist_size, position=[0.81, 0.1, 0.6], color=format_color(col)))
        objects = [Shape(target_name)] + distrators
        objects[0].set_color(format_color(target_color))
        results_lists = multiple_tests(n_tests, model, env, camera, arm, objects, target_dummy, ref, max_steps, distance_tolerance, angle_tolerance, maintain_target_duration)
        for dist in distrators:
            dist.remove()

        is_reached_list, n_steps_list = results_lists
        sucess_results.append(is_reached_list)
        steps_needed.append(n_steps_list)

        print(f'All test done! Final accuracy:')
        print(sum(is_reached_list)/len(is_reached_list))
        save_results(save_loc, model_names, sucess_results, steps_needed, i)
        print("Saved for scale:", model_names)

    env.shutdown()

if __name__ == "__main__":
    main()
    print("All done, all saved!")