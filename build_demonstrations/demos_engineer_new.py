import numpy as np
import warnings
from pyrep.errors import IKError
from pyrep.objects.dummy import Dummy
from demos_control import Controller
from scipy.spatial.transform import Rotation


class NotReachedError(Exception):
    """When the robot fails to reach a target"""


class Trajectory():
    def __init__(self):
        self.image_seq = []
        self.action_seq = []
        self.robot_state_seq = []
        self.control_inp_seq = []
        self.object_pos_seq = []

    def __len__(self):
        return len(action_seq)

    def add(self, image, action, robot_state, control_input, object_pose):
        self.image_seq.append(image)
        self.action_seq.append(action)
        self.robot_state_seq.append(robot_state)
        self.control_inp_seq.append(control_input)
        self.object_pos_seq.append(object_pose)

    def get_as_arrays(self):
        all_seq = [self.image_seq, self.action_seq, self.robot_state_seq, self.control_inp_seq, self.object_pos_seq,]
        all_seq_as_arrays = list(map(np.array, all_seq))
        return all_seq_as_arrays

class Demonstartions():
    def __init__(self):
        self.trajectories = []
        self.demo_image_sequences = []
        self.demo_action_sequences = []
        self.demo_robot_state_sequences = []
        self.demo_control_input_sequences = []
        self.demo_target_poses_sequences = []
        self.init_obj_poses = []
        self.init_robot_joints = []
        self.n_steps_per_traj = []

    def add_seq(self, trajectory):
        n_steps = len(trajectory)
        self.trajectories.append(trajectory)

    def save(self, location):
        demo_arrays = self.get_as_dict_of_arrays()
        np.savez(location, **demo_arrays)



def get_rot_towards_target(current, target, ref):
    current_pos = current.get_position(relative_to=ref)
    target_pos = target.get_position(relative_to=ref)
    # v for vector
    v_towards_target = target_pos - current_pos
    v_towards_target /= np.linalg.norm(v_towards_target)
    v_down = np.array([0, 0, -1])
    v_halfway = v_towards_target + v_down
    v_halfway /= np.linalg.norm(v_halfway)
    cross_product = np.cross(v_down, v_halfway)
    dot_product = [np.dot(v_towards_target, v_halfway),]
    # cross_product = qx, qy, qz; dot_product = qw
    quat_towards_target = np.concatenate([cross_product, dot_product])
    rot_towards_target = Rotation.from_quat(quat_towards_target)
    rot_target = Rotation.from_quat(target.get_quaternion(relative_to=ref))
    rot_total = rot_towards_target * rot_target

    return rot_total


def demo_reach_trajectory(env, arm, camera, target_obj, t_dummy, ref, max_steps, max_speed_linear, max_speed_angular, precision_linear, precision_angular, maintain):
    """Collectect demo images and actions for reaching a target dummy pose"""
    traj = Trajectory()

    controller = Controller(env, arm, t_dummy, ref)
    step_counter = 0
    error_counter = 0
    maintain_good_pose_counter = 0

    # Step 1: reach pose1 5cm above target pointing down towards the target
    target1_position = target_obj.get_position(relative_to=ref) + np.array([0, 0, 0.05])
    rot_towards_target = get_rot_towards_target(arm.get_tip(), target_obj, ref)
    target1_pose = np.concatenate([target1_position, rot_towards_target.as_quat()])
    t_dummy.set_pose(target1_pose, relative_to=ref)

    dist_pos, dist_ori = controller.get_distances()

    while dist_pos > precision_linear or dist_ori > precision_angular:
        try:
            delta_pos, delta_quat, next_pos, next_quat = controller.get_pose_step(max_speed_linear, max_speed_angular)
        except IKError as e:
            error_counter += 1
            if error_counter < 10:
                continue
            raise e

        # Capture image and do action
        img = camera.capture_rgb()
        robot_state, controls = controller.demo_step(next_pos, next_quat)

        # Record the state action + other context info
        traj.add(img, np.concatenate([delta_pos, delta_quat]), robot_state, controls, target_obj.get_pose(relative_to=arm.get_tip()))

        # Update target orientation so that camera points towards target
        rot_towards_target = get_rot_towards_target(arm.get_tip(), target_obj, ref)
        target1_pose = np.concatenate([target1_position, rot_towards_target.as_quat()])
        t_dummy.set_pose(target1_pose, relative_to=ref)

        step_counter += 1
        dist_pos, dist_ori = controller.get_distances()

        if step_counter > max_steps:
            raise NotReachedError("Max steps per trajectory")

        inp = input()
        if inp == 'q':
            env.shutdown()

    # Step 2: reach down, so that target lies between the 2 fingers of gripper
    target_pose3 = target_obj.get_pose(relative_to=ref)
    t_dummy.set_pose(target_pose3, relative_to=ref)

    while dist_pos > precision_linear or dist_ori > precision_angular or maintain_good_pose_counter < maintain:
        if dist_pos < precision_linear and dist_ori < precision_angular:
            maintain_good_pose_counter += 1
        else:
            maintain_good_pose_counter = 0
        try:
            delta_pos, delta_quat, next_pos, next_quat = controller.get_pose_step(max_speed_linear, max_speed_angular)
        except IKError as e:
            error_counter += 1
            if error_counter < 10:
                continue
            raise e

        # Capture image and do action
        img = camera.capture_rgb()
        robot_state, controls = controller.demo_step(next_pos, next_quat)

        # Record the state action + other context info
        traj.add(img, np.concatenate([delta_pos, delta_quat]), robot_state, controls, target_obj.get_pose(relative_to=arm.get_tip()))

        step_counter += 1
        dist_pos, dist_ori = controller.get_distances()

        if step_counter > max_steps:
            raise NotReachedError("Max steps per trajectory")

        inp = input()
        if inp == 'q':
            env.shutdown()

    return traj.get_as_arrays()


def reset_scene(init_obj_poses, init_robot_joints, arm, target_obj, distractors, ref):
    all_obj = [target_obj] + distractors
    for i_obj, obj in enumerate(all_obj):
        obj.set_position(init_obj_poses[i_obj, :3], relative_to=ref)
        obj.set_orientation(init_obj_poses[i_obj, 3:], relative_to=ref)
    arm.set_joint_positions(init_robot_joints, disable_dynamics=True)


def save(data_dict, location):
    print("Compressing and saving to", location)
    for key, value in data_dict.items():
        if key[:4] == 'demo':
            data_dict[key] = np.concatenate(value)
        else:
            data_dict[key] = np.array(value)
    np.savez_compressed(location, **data_dict)


    
def collect_and_save_demos(env, arm, camera, target_obj, target_dummy, distractors, ref, n_demos, max_steps, init_config_file, save_demo_location, max_speed_linear, max_speed_angular, precision_linear, precision_angular, maintain):
    """collect a set of demonstration trajectories"""
    arm.set_control_loop_enabled(False)
    
    init_config = np.load(init_config_file, allow_pickle=True)
    init_obj_poses = init_config['obj_poses']
    init_robot_joints = init_config['joint_angles']
    max_demos = len(init_robot_joints)
    if n_demos is None:
        n_demos = max_demos
    
    counter_demo_done = 0
    
    data = {'demo_image_sequences': [],
            'demo_action_sequences':[],
            'demo_robot_state_sequences': [],
            'demo_control_input_sequences': [],
            'demo_target_poses_sequences': [],
            'init_obj_poses': [],
            'init_robot_joints': [],
            'n_steps_per_traj': [],}

    for itr, (obj_poses, robot_joints) in enumerate(zip(init_obj_poses, init_robot_joints)):
        env.stop()
        env.start()
        print(f"Demo setup: {itr+1}/{max_demos}")
        reset_scene(obj_poses, robot_joints, arm, target_obj, distractors, ref)
        # print("scene resetted")
        # target_dummy = create_dummy_target(target_obj, ref)

        try:
            demo_traj_seq = demo_reach_trajectory(env, arm, camera, target_obj, target_dummy, ref, max_steps, max_speed_linear, max_speed_angular, precision_linear, precision_angular, maintain)
        except (NotReachedError, IKError) as e:
            warnings.warn(str(e), Warning)
            continue
        env.stop()
        data['demo_image_sequences'].append(demo_traj_seq[0])
        data['demo_action_sequences'].append(demo_traj_seq[1])
        data['demo_robot_state_sequences'].append(demo_traj_seq[2])
        data['demo_control_input_sequences'].append(demo_traj_seq[3])
        data['demo_target_poses_sequences'].append(demo_traj_seq[4])
        data['init_obj_poses'].append(obj_poses)
        data['init_robot_joints'].append(robot_joints)
        data['n_steps_per_traj'].append(len(demo_traj_seq[1]))

        counter_demo_done += 1
        print(f"Demos collected {counter_demo_done}/{n_demos}")
        if counter_demo_done == n_demos:
            break

        # if counter_demo_done%100 == 0:
        #     save(data, save_demo_location)
        #     print(f'Collected and saved {counter_demo_done} demos!')
    env.shutdown()
    save(data, save_demo_location)
    print(f"Total: {counter_demo_done} demos successfully collected!")
