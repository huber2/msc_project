from os.path import dirname, abspath
from datetime import datetime
import numpy as np
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.dummy import Dummy
from scipy.spatial.transform import Rotation
from pyrep.errors import ConfigurationError


def get_scene_random_initial_positions(n_demos, n_objects):

    # Robotic arm tip
    tip_bounding_box = [-0.2, -0.2, 0.35, 0.2, 0.2, 0.7]
    tip_bounding_euler_angles = [-60, -30, -30, 60, 30, 30]
    tip_min = np.concatenate([tip_bounding_box[:3], tip_bounding_euler_angles[:3]]).reshape(1, 6)
    tip_max = np.concatenate([tip_bounding_box[3:], tip_bounding_euler_angles[3:]]).reshape(1, 6)
    tip_random_init_poses = np.random.uniform(tip_min, tip_max, (n_demos, 6))
    
    # Objects
    obj_bounding_box = [0.2, -0.3, 0.025, 0.8, 0.3, 0.025]
    obj_bounding_euler_angles = np.array([180, 0, -180, 180, 0, 180]) * np.pi/180
    obj_min = np.concatenate([obj_bounding_box[:3], obj_bounding_euler_angles[:3]]).reshape(1, 6)
    obj_max = np.concatenate([obj_bounding_box[3:], obj_bounding_euler_angles[3:]]).reshape(1, 6)
    all_obj_init_poses = np.random.uniform(obj_min, obj_max, (n_demos, n_objects, 6))

    return tip_random_init_poses, all_obj_init_poses


def set_init_poses(env, arm, target_obj, distractors, camera, ref, n_demos, tip_random_init_poses, all_obj_init_poses, max_errors):
    good_init_idx = []
    good_robot_joints_configs = []
    init_imgs = []
    all_obj = [target_obj] + distractors
    counter_errors = 0
    for i_dem in range(n_demos):
        if i_dem % 50 == 0:
            print('i_dem', i_dem)
        env.start()
        for i_obj, obj in enumerate(all_obj):
            obj.set_position(all_obj_init_poses[i_dem, i_obj, :3], relative_to=ref)
            obj.set_orientation(all_obj_init_poses[i_dem, i_obj, 3:], relative_to=ref)
        
        # check collisions
        is_collision = False
        for i in range(1, len(all_obj)):
            for j in range(i):
                if all_obj[i].check_collision(all_obj[j]):
                    is_collision = True
        if is_collision:
            continue


        
        target_pos = target_obj.get_position(relative_to=ref)
        tip_pos = tip_random_init_poses[i_dem, :3]
        tip_pos[:2] += target_pos[:2]
        #angles_towards_target = get_orientation_towards_target(tip_pos, target_pos)
        #target_rot = Rotation.from_euler('xyz', angles_towards_target, degrees=False)
        target_rot = Rotation.from_euler('xyz', [0, 180, 0], degrees=True)
        random_rot = Rotation.from_euler('ZYX', tip_random_init_poses[i_dem, 3:], degrees=True)
        total_rot = target_rot * random_rot

        try:
            q = arm.get_configs_for_tip_pose(tip_pos, quaternion=total_rot.as_quat(), relative_to=ref)[0]
        except ConfigurationError as e:
            counter_errors += 1
            if counter_errors < max_errors:
                continue
            raise e
        arm.set_joint_positions(q, disable_dynamics=True)

        env.step()
        img = camera.capture_rgb()
        init_imgs.append(img)
        good_robot_joints_configs.append(arm.get_joint_positions())
        good_init_idx.append(i_dem)
        env.stop()
    
    return np.array(good_init_idx), np.array(good_robot_joints_configs), np.array(init_imgs)

        


def get_orientation_towards_target(tip_position, target_position):
    delta = target_position - tip_position
    dist = np.linalg.norm(delta)
    rot_y = np.pi - np.arccos(-delta[2]/dist)
    if delta[0] == 0:
        rot_z = np.sign(delta[1]) * np.pi/2
    elif delta[0] > 0:
        rot_z = np.arctan(delta[1]/delta[0])
    else:
        rot_z = np.pi + np.arctan(delta[1]/delta[0])
    return np.array([0, rot_y, rot_z])



DIR_PATH = dirname(abspath(__file__)) + '/../'
SCENE_FILE = DIR_PATH + 'coppelia_scenes/Franka_bc.ttt'
timestamp = datetime.now().strftime('%y_%m_%d_%_H_%M_%S')
save_location = DIR_PATH + 'data/scene_setups_' + timestamp

n_demos = 5
max_errors = 20

env = PyRep()
env.launch(SCENE_FILE, headless=True, responsive_ui=False)
arm = Panda()
object_names = ['blue_target', 'red_cube', 'red_cube_big', 'lime_cylinder', 'orange_sphere']
objects = [Shape(obj) for obj in object_names]
target_object = objects[0]
distractors = objects[1:]
n_objects = len(object_names)
camera = VisionSensor('Panda_camera')
ref = arm.get_object('Reference')

tip_random_init_poses, all_obj_init_poses = get_scene_random_initial_positions(n_demos, n_objects)

good_init_idx, good_robot_joints_configs, init_imgs = set_init_poses(env, arm, target_object, distractors, camera, ref, n_demos, tip_random_init_poses, all_obj_init_poses, max_errors)
print('good init found', len(good_init_idx))
env.shutdown()

good_init_configs = {}
good_init_configs["joint_angles"] = good_robot_joints_configs
good_init_configs["obj_poses"] = all_obj_init_poses[good_init_idx]
good_init_configs["images"] = init_imgs

np.savez_compressed(save_location, **good_init_configs)

