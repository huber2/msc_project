import numpy as np
import warnings
from pyrep.errors import IKError
from demos_control_new import Controller
from scipy.spatial.transform import Rotation


class NotReachedError(Exception):
    """When the robot fails to reach a target"""


class Trajectory:
    def __init__(self):
        self.image_seq = []
        self.action_seq = []
        self.robot_state_seq = []
        self.control_inp_seq = []
        self.target_pose_seq = []

    def __len__(self):
        return len(self.action_seq)

    def add(self, image, action, robot_state, control_input, target_pose):
        self.image_seq.append(image)
        self.action_seq.append(action)
        self.robot_state_seq.append(robot_state)
        self.control_inp_seq.append(control_input)
        self.target_pose_seq.append(target_pose)

    def get_as_dict_of_arrays(self):
        as_dict_of_arrays = {k: np.array(v) for (k, v) in self.__dict__.items()}
        return as_dict_of_arrays


class Demonstrations:
    def __init__(self):
        self.traj_data_naming = {
            'demo_image_sequences': 'image_seq',
            'demo_action_sequences': 'action_seq',
            'demo_robot_state_sequences': 'robot_state_seq',
            'demo_control_input_sequences': 'control_inp_seq',
            'demo_target_poses_sequences': 'target_pose_seq',
        }
        self.count_steps = 0
        self.data = {'total_steps': [],
                     'init_obj_poses': [],
                     'init_robot_joints': [],}
        for key in self.traj_data_naming:
            self.data[key] = []

    def add_trajectory(self, trajectory, init_obj_poses, init_robot_joints):
        self.count_steps += len(trajectory)
        self.data['total_steps'].append(self.count_steps)
        self.data['init_obj_poses'].append(init_obj_poses)
        self.data['init_robot_joints'].append(init_robot_joints)
        traj = trajectory.get_as_dict_of_arrays()
        for (key, value) in self.traj_data_naming.items():
            self.data[key].append(traj[value])

    def get_as_dict_arrays(self):
        dict_of_arrays = {k: np.concatenate(self.data[k]) for k in self.traj_data_naming}
        for k in ['total_steps', 'init_obj_poses', 'init_robot_joints']:
            dict_of_arrays[k] = np.array(self.data[k])
        return dict_of_arrays

    def save(self, location):
        np.savez(location, **self.get_as_dict_arrays())


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
    """Collect demo images and actions for reaching a target dummy pose"""
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

    dist_pos, dist_ori = controller.get_distance_linear(), controller.get_distance_angular()

    while dist_pos > precision_linear or dist_ori > precision_angular:
        # Capture image and do action
        img = camera.capture_rgb()
        try:
            v_linear, v_angular = controller.step_with_velocities(max_speed_linear, max_speed_angular)
        except IKError as e:
            error_counter += 1
            if error_counter < 10:
                continue
            raise e

        # Record the state action + other context info
        act = np.concatenate([v_linear, v_angular])
        robot_state = arm.get_joint_positions()
        controls = arm.get_joint_target_velocities()
        target_pose = target_obj.get_pose(retative_to= arm.get_tip())
        traj.add(img, act, robot_state, controls, target_pose)

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

        # Capture image and do action
        img = camera.capture_rgb()
        try:
            v_linear, v_angular = controller.step_with_velocities(
                max_speed_linear, max_speed_angular)
        except IKError as e:
            error_counter += 1
            if error_counter < 10:
                continue
            raise e

        # Record the state action + other context info
        act = np.concatenate([v_linear, v_angular])
        robot_state = arm.get_joint_positions()
        controls = arm.get_joint_target_velocities()
        target_pose = target_obj.get_pose(retative_to=arm.get_tip())
        traj.add(img, act, robot_state, controls, target_pose)
        step_counter += 1
        dist_pos, dist_ori = controller.get_distance_linear(), controller.get_distance_angular()

        if step_counter > max_steps:
            raise NotReachedError("Max steps per trajectory")

        inp = input()
        if inp == 'q':
            env.shutdown()

    return traj


def reset_scene(init_obj_poses, init_robot_joints, arm, target_obj, distractors, ref):
    all_obj = [target_obj] + distractors
    for i_obj, obj in enumerate(all_obj):
        obj.set_position(init_obj_poses[i_obj, :3], relative_to=ref)
        obj.set_orientation(init_obj_poses[i_obj, 3:], relative_to=ref)
    arm.set_joint_positions(init_robot_joints, disable_dynamics=True)


def collect_and_save_demos(env, arm, camera, target_obj, target_dummy, distractors, ref, n_demos, max_steps, init_config_file, save_demo_location, max_speed_linear, max_speed_angular, precision_linear, precision_angular, maintain):
    """collect a set of demonstration trajectories"""
    arm.set_control_loop_enabled(False)
    
    init_config = np.load(init_config_file)
    init_obj_poses = init_config['obj_poses']
    init_robot_joints = init_config['joint_angles']
    max_demos = len(init_robot_joints)
    if n_demos is None:
        n_demos = max_demos
    
    counter_demo_done = 0
    
    demos = Demonstrations()

    for itr, (obj_poses, robot_joints) in enumerate(zip(init_obj_poses, init_robot_joints)):
        env.stop()
        env.start()
        print(f"Demo setup: {itr+1}/{max_demos}")
        reset_scene(obj_poses, robot_joints, arm, target_obj, distractors, ref)

        try:
            traj = demo_reach_trajectory(env, arm, camera, target_obj, target_dummy, ref, max_steps, max_speed_linear, max_speed_angular, precision_linear, precision_angular, maintain)
        except (NotReachedError, IKError) as e:
            warnings.warn(str(e), Warning)
            continue
        env.stop()
        demos.add_trajectory(traj, obj_poses, robot_joints)
        counter_demo_done += 1
        print(f"Demos collected {counter_demo_done}/{n_demos}")
        if counter_demo_done == n_demos:
            break

        if counter_demo_done%100 == 0:
            demos.save(save_demo_location)
            print(f'Collected and saved {counter_demo_done} demos!')
    env.shutdown()
    demos.save(save_demo_location)
    print(f"Total: {counter_demo_done} demos successfully collected!")
