import numpy as np
import warnings
from pyrep.errors import IKError
from demos_control_new import Controller


class NotReachedError(Exception):
    """When the robot fails to reach a target"""


class Trajectory:
    def __init__(self):
        self.image_seq = []
        self.action_seq = []

    def __len__(self):
        return len(self.action_seq)

    def add(self, image, action):
        self.image_seq.append(image)
        self.action_seq.append(action)


class Demonstrations:
    def __init__(self):
        self.count_steps = 0
        self.step_marker = []
        self.demo_image_sequences = []
        self.demo_action_sequences = []

    def add_trajectory(self, trajectory):
        self.count_steps += len(trajectory)
        self.step_marker.append(self.count_steps)
        self.demo_image_sequences.append(np.array(trajectory.image_seq))
        self.demo_action_sequences.append(np.array(trajectory.action_seq))
        
    def get_as_dict_of_arrays(self):
        dict_of_arrays = {
            'demo_image_sequences': np.concatenate(self.demo_image_sequences),
            'demo_action_sequences': np.concatenate(self.demo_action_sequences),
            'step_marker': np.array(self.step_marker) 
        }
        return dict_of_arrays
        
    def save(self, location):
        np.savez(location, **self.get_as_dict_of_arrays())

        
class Controller:

    def __init__(self, env, arm, target_dummy, reference_dummy):
        self.env = env
        self.arm = arm
        self.tgt = target_dummy
        self.ref = reference_dummy
         
    def get_dist_linear(self):
        # Position error
        tgt_pos = np.array(self.tgt.get_position(relative_to=self.ref))
        tip_pos = np.array(self.arm.get_tip().get_position(relative_to=self.ref))
        diff_pos = tgt_pos - tip_pos
        dist_pos = np.linalg.norm(diff_pos)
        return dist_pos
    
    def get_tip_linear_displacement(self, max_speed_linear):
        tgt_pos = np.array(self.tgt.get_position(relative_to=self.ref))
        tip_pos = np.array(self.arm.get_tip().get_position(relative_to=self.ref))
        diff_pos = tgt_pos - tip_pos
        dist_pos = np.linalg.norm(diff_pos)
        whole_step_speed_linear = dist_pos / self.env.get_simulation_timestep()
        coef_step_linear = min(1, max_speed_linear/whole_step_speed_linear)
        delta_pos = diff_pos * coef_step_linear
        next_pos = tip_pos + delta_pos
        return delta_pos, next_pos

    def get_joint_velocities_from_tip_displacement_with_ik_solver(self, next_tip_pos, next_tip_quat):
        next_joint_angles = self.arm.solve_ik_via_jacobian(next_tip_pos, quaternion=next_tip_quat, relative_to=self.ref)
        current_joint_angles = np.array(self.arm.get_joint_positions())
        v_joints = (next_joint_angles - current_joint_angles) / self.env.get_simulation_timestep()
        return v_joints
    
    def step_with_displacement(self, max_speed_linear):
        delta_pos, next_pos = self.get_tip_linear_displacement(max_speed_linear)
        next_quat = self.tgt.get_quaternion(relative_to=self.ref)
        v_joints = self.get_joint_velocities_from_tip_displacement_with_ik_solver(next_pos, next_quat)
        self.arm.set_joint_target_velocities(v_joints)
        self.env.step()
        return delta_pos


def get_random_initial_poses(n_demos):
    obj_bounding_box = [0.35, -0.4, 0.03, 0.85, 0.4, 0.03]
    obj_bounding_euler_angles = np.array([180, 0, -180, 180, 0, 180]) * np.pi/180
    obj_min = np.concatenate([obj_bounding_box[:3], obj_bounding_euler_angles[:3]]).reshape(1, 6)
    obj_max = np.concatenate([obj_bounding_box[3:], obj_bounding_euler_angles[3:]]).reshape(1, 6)
    obj_init_poses = np.random.uniform(obj_min, obj_max, (n_demos, 6))
    return obj_init_poses


def demo_reach_trajectory(env, arm, camera, target_obj, t_dummy, ref, max_steps, max_speed_linear, precision_linear, maintain, headless):
    """Collect demo images and actions for reaching a target dummy pose"""
    traj = Trajectory()

    controller = Controller(env, arm, t_dummy, ref)
    step_counter = 0
    error_counter = 0
    maintain_good_pose_counter = 0

    # Reach pose 5cm above target
    target1_position = np.array(target_obj.get_position(relative_to=ref)) + np.array([0, 0, 0.05])
    target1_quat = np.array(arm.get_tip().get_quaternion(relative_to=ref))
    target1_pose = np.concatenate([target1_position, target1_quat])
    t_dummy.set_pose(target1_pose, relative_to=ref)

    dist_pos = controller.get_dist_linear()

    while dist_pos > precision_linear or maintain_good_pose_counter < maintain:
        # Capture image and do action
        img = camera.capture_rgb()
        if dist_pos < precision_linear:
            maintain_good_pose_counter += 1
        else:
            maintain_good_pose_counter = 0
        try:
            v_linear = controller.step_with_displacement(max_speed_linear)
        except IKError as e:
            warnings.warn(str(e), Warning)
            error_counter += 1
            if error_counter < 10:
                env.step()
                continue
            raise e

        # Record the state action + other context info
        traj.add(img, v_linear)

        step_counter += 1
        dist_pos = controller.get_dist_linear()

        if step_counter > max_steps:
            raise NotReachedError("Max steps per trajectory")

        print('step', step_counter, end=' ')
        if not headless:
            inp = input()
            if inp == 'q':
                env.shutdown()

    return traj


def collect_and_save_demos(env, arm, camera, target_obj, target_dummy, ref, n_demos, max_steps, save_demo_location, max_speed_linear, precision_linear, maintain, headless):
    """collect a set of demonstration trajectories"""
    arm.set_control_loop_enabled(False)
    counter_demo_done = 0
    demos = Demonstrations()

    init_pos = get_random_initial_poses(n_demos)

    for itr in range(n_demos):
        env.stop()
        env.start()
        target_obj.set_position(init_pos[itr, :3], relative_to=ref)
        target_obj.set_orientation(init_pos[itr, 3:], relative_to=ref)
        env.step()
        print(f"Demo setup: {itr+1}/{n_demos}")
        try:
            traj = demo_reach_trajectory(env, arm, camera, target_obj, target_dummy, ref, max_steps, max_speed_linear, precision_linear, maintain, headless)
        except (NotReachedError, IKError) as e:
            warnings.warn('10X '+str(e), Warning)
            continue
        env.stop()
        demos.add_trajectory(traj)
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

