from os.path import dirname, abspath
from datetime import datetime
import numpy as np
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.objects.shape import Shape


DIR_PATH = dirname(abspath(__file__)) + '/../'
SCENE_FILE = DIR_PATH + 'coppelia_scenes/Franka_bc.ttt'
timestamp = datetime.now().strftime('%y_%m_%d_%_H_%M_%S')
save_location = DIR_PATH + 'data/scene_setups_' + timestamp

n_config_tries = 300
env = PyRep()

init_robot_joints = np.array([
    [0, 15, 0, -70, 0, 85, 45],
    [45, 15, 0, -70, -30, 85, 0],
    [-45, 15, 0, -70, +30, 85, 0],
]) * np.pi / 180


def get_random_initial_poses(n_demos, n_objects):
    obj_bounding_box = [0.3, -0.2, 0.025, 0.8, 0.2, 0.025]
    obj_bounding_euler_angles = np.array([180, 0, -180, 180, 0, 180]) * np.pi/180
    obj_min = np.concatenate([obj_bounding_box[:3], obj_bounding_euler_angles[:3]]).reshape(1, 6)
    obj_max = np.concatenate([obj_bounding_box[3:], obj_bounding_euler_angles[3:]]).reshape(1, 6)
    obj_init_poses = np.random.uniform(obj_min, obj_max, (n_demos, n_objects, 6))

    return obj_init_poses


def get_objects_init_poses_no_collisions(env, scene, n_demos):
    env.launch(scene, headless=True, responsive_ui=False)
    arm = Panda()
    ref = arm.get_object('Reference')
    objects_names = ['blue_target', 'red_cube', 'red_cube_big', 'lime_cylinder', 'orange_sphere']
    objects = [Shape(obj) for obj in objects_names]
    configs_no_collision = []
    init_obj_poses = get_random_initial_poses(n_demos, len(objects))
    env.start()
    for poses in init_obj_poses:
        for i_obj, obj in enumerate(objects):
            obj.set_position(poses[i_obj, :3], relative_to=ref)
            obj.set_orientation(poses[i_obj, 3:], relative_to=ref)
        is_collision = False
        for i in range(1, len(objects)):
            for j in range(i):
                if objects[i].check_collision(objects[j]):
                    is_collision = True
        if not is_collision:
            configs_no_collision.append(poses)
    env.stop()
    env.shutdown()
    print(f"No collisions in {len(configs_no_collision)}/{n_demos}) cases")
    return np.stack(configs_no_collision)


def save_init_configs(env, scene, n_demos, init_robot_joints, save_location):
    n_demos_per_robot_config = n_demos // len(init_robot_joints) + 1
    init_obj_poses = get_objects_init_poses_no_collisions(env, scene, n_demos_per_robot_config)
    a, b = np.mgrid[:len(init_robot_joints), :len(init_obj_poses)]
    init_configs = {
        "obj_poses": np.concatenate(init_obj_poses[b]),
        "joint_angles": np.concatenate(init_robot_joints[a])}
    np.savez_compressed(save_location, **init_configs)


save_init_configs(env, SCENE_FILE, n_config_tries, init_robot_joints, save_location)