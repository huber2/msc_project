import numpy as np
import warnings
from pyrep.errors import IKError
from pyrep.objects.dummy import Dummy
from demos_control import Controller
from scipy.spatial.transform import Rotation


class NotReachedError(Exception):
    """When the robot fails to reach a target"""


def demo_reach_pose(env, arm, camera, t_dummy, ref, max_steps, max_speed_pos, max_speed_angular, precision_linear, precision_angular, maintain):
    """Collectect demo images and actions for reaching a target dummy pose"""
    image_seq = []
    action_seq = []
    robot_state_seq = []
    control_inp_seq = []
    controller = Controller(env, arm, t_dummy, ref)
    step_counter = 0
    error_counter = 0
    maintain_good_pose_counter = 0
    dist_pos, dist_ori = controller.get_distances()
    while dist_pos > precision_linear or dist_ori > precision_angular or maintain_good_pose_counter < maintain:
        # print(f"step #{step_counter}")
        if (dist_pos < precision_linear and dist_ori < precision_angular):
            maintain_good_pose_counter += 1
        else:
            maintain_good_pose_counter = 0
        try:
            delta_pos, delta_quat, next_pos, next_quat = controller.get_pose_step(max_speed_pos, max_speed_angular)
        except IKError as e:
            error_counter += 1
            if error_counter < 10:
                continue
            raise e
        img = camera.capture_rgb()
        robot_state, controls = controller.demo_step(next_pos, next_quat)
        image_seq.append(img)
        action_seq.append(np.concatenate([delta_pos, delta_quat]))
        robot_state_seq.append(robot_state)
        control_inp_seq.append(controls)
        step_counter += 1
        dist_pos, dist_ori = controller.get_distances()
        # inp = input()
        # if inp == 'q':
        #     env.shutdown()
        if step_counter > max_steps:
            raise NotReachedError("Max steps per trajectory")
    return image_seq, action_seq, robot_state_seq, control_inp_seq


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

def demo_trajectory(env, arm, camera, target_obj, target_dummy, ref, max_steps,  max_speed_linear, max_speed_angular, precision_linear, precision_angular, maintain):
    """Collect one full demonstration trajectory"""
    image_seq = []
    action_seq = []
    robot_state_seq = []
    control_inp_seq = []
    
    # Compute the rotation such that the camera points its z axis towards the object
    rot_towards_target = get_rot_towards_target(camera, target_obj, ref)

    # First part is to make the tip point towards target and go towards the target in a straight line
    target1_position = target_obj.get_position(relative_to=ref) + np.array([0, 0, 0.05])
    target1_pose = np.concatenate([target1_position, rot_towards_target.as_quat()])
    target_dummy.set_pose(target1_pose, relative_to=ref)

    imgs1, acts1, states1, controls1 = demo_reach_pose(env, arm, camera, target_dummy, ref, max_steps, max_speed_linear, max_speed_angular, precision_linear, precision_angular, 0)
    image_seq += imgs1
    action_seq += acts1
    robot_state_seq += states1
    control_inp_seq += controls1
    
    #Then, once the target is reached, orient the gripper correctly
    target_quat = target_obj.get_quaternion(relative_to=ref)
    target_pose2 = np.concatenate([target1_position, target_quat])
    target_dummy.set_pose(target_pose2, relative_to=ref)
    imgs2, acts2, states2, controls2 = demo_reach_pose(env, arm, camera, target_dummy, ref, max_steps, max_speed_linear, max_speed_angular, precision_linear, precision_angular, 0)
    image_seq += imgs2
    action_seq += acts2
    robot_state_seq += states2
    control_inp_seq += controls2

    #Finally, move down towards the object
    target_pose3 = target_obj.get_pose(relative_to=ref)
    target_dummy.set_pose(target_pose3, relative_to=ref)
    imgs2, acts2, states2, controls2 = demo_reach_pose(env, arm, camera, target_dummy, ref, max_steps, max_speed_linear, max_speed_angular, precision_linear, precision_angular, maintain)
    image_seq += imgs2
    action_seq += acts2
    robot_state_seq += states2
    control_inp_seq += controls2
    
    return np.array(image_seq), np.array(action_seq), np.array(robot_state_seq), np.array(control_inp_seq)



def reset_scene(init_obj_poses, init_robot_joints, arm, target_obj, distractors, ref):
    all_obj = [target_obj] + distractors
    for i_obj, obj in enumerate(all_obj):
        obj.set_position(init_obj_poses[i_obj, :3], relative_to=ref)
        obj.set_orientation(init_obj_poses[i_obj, 3:], relative_to=ref)
    arm.set_joint_positions(init_robot_joints, disable_dynamics=True)

    
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
            'init_obj_poses': [],
            'init_robot_joints': [],}

    for itr, (obj_poses, robot_joints) in enumerate(zip(init_obj_poses, init_robot_joints)):
        env.stop()
        env.start()
        print(f"Demo setup: {itr+1}/{max_demos}")
        print('...env started')
        reset_scene(obj_poses, robot_joints, arm, target_obj, distractors, ref)
        # print("scene resetted")
        # target_dummy = create_dummy_target(target_obj, ref)

        try:
            demo_traj_seq = demo_trajectory(env, arm, camera, target_obj, target_dummy, ref, max_steps, max_speed_linear, max_speed_angular, precision_linear, precision_angular, maintain)
        except (NotReachedError, IKError) as e:
            warnings.warn(str(e), Warning)
            continue
        finally:
            
            pass

        data['demo_image_sequences'].append(demo_traj_seq[0])
        data['demo_action_sequences'].append(demo_traj_seq[1])
        data['demo_robot_state_sequences'].append(demo_traj_seq[2])
        data['demo_control_input_sequences'].append(demo_traj_seq[3])
        data['init_obj_poses'].append(obj_poses)
        data['init_robot_joints'].append(robot_joints)

        counter_demo_done += 1
        print(f"{counter_demo_done}/{n_demos} demos collected")
        if counter_demo_done == n_demos:
            break

        if counter_demo_done%100 == 0:
            np.savez_compressed(save_demo_location, **data)
            print(f'Collected and saved {counter_demo_done} demos!')

    np.savez_compressed(save_demo_location, **data)
    print(f"Total: {counter_demo_done} demos successfully collected!")
