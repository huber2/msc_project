import numpy as np
import warnings
from pyrep.errors import IKError
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp


class NotReachedError(Exception):
    """When the robot fails to reach a target"""


class Trajectory:
    def __init__(self):
        self.image_seq = []
        self.action_seq = []

    def __len__(self):
        return len(self.action_seq)

    def add(self, image, action):
        image_8bit = np.array(image * 255, dtype=np.uint8)
        self.image_seq.append(image_8bit)
        self.action_seq.append(action)


class Demonstrations:
    def __init__(self):
        self.count_steps = 0
        self.step_marker = [0]
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
        np.savez_compressed(location, **self.get_as_dict_of_arrays())

        
class Controller:

    def __init__(self, env, arm, target_dummy, reference_dummy):
        self.env = env
        self.arm = arm
        self.tgt = target_dummy
        self.ref = reference_dummy
         
    def get_dist_angular(self):
        tgt_quat = self.tgt.get_quaternion(relative_to=self.ref)
        tip_quat = self.arm.get_tip().get_quaternion(relative_to=self.ref)
        tip_rot = Rotation.from_quat(tip_quat)
        tgt_rot = Rotation.from_quat(tgt_quat)
        diff_rot = tgt_rot * tip_rot.inv()
        dist_angle = diff_rot.magnitude()
        return dist_angle

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

    def get_tip_angular_displacement(self, max_speed_angular):
        # Orientation error
        tgt_quat = self.tgt.get_quaternion(relative_to=self.ref)
        tip_quat = self.arm.get_tip().get_quaternion(relative_to=self.ref)
        tip_rot = Rotation.from_quat(tip_quat)
        tgt_rot = Rotation.from_quat(tgt_quat)
        diff_rot = tgt_rot * tip_rot.inv()
        dist_angle = diff_rot.magnitude()
        # Angular displacement
        whole_step_speed_angular = dist_angle / self.env.get_simulation_timestep()
        coef_step_angular = min(1, max_speed_angular/whole_step_speed_angular)
        key_rotations = Rotation.concatenate([tip_rot, tgt_rot])
        key_times = [0, 1]
        slerp = Slerp(key_times, key_rotations)
        next_rot = slerp(coef_step_angular)
        next_quat = next_rot.as_quat()
        delta_rot = next_rot * tip_rot.inv()
        return delta_rot, next_quat

    def get_joint_velocities_from_tip_displacement_with_ik_solver(self, next_tip_pos, next_tip_quat):
        next_joint_angles = self.arm.solve_ik_via_jacobian(next_tip_pos, quaternion=next_tip_quat, relative_to=self.ref)
        current_joint_angles = np.array(self.arm.get_joint_positions())
        v_joints = (next_joint_angles - current_joint_angles) / self.env.get_simulation_timestep()
        return v_joints
    
    def step_with_displacement(self, max_speed_linear, max_speed_angular):
        delta_pos, next_pos = self.get_tip_linear_displacement(max_speed_linear)
        delta_rot, next_quat = self.get_tip_angular_displacement(max_speed_angular)
        v_joints = self.get_joint_velocities_from_tip_displacement_with_ik_solver(next_pos, next_quat)
        self.arm.set_joint_target_velocities(v_joints)
        self.env.step()
        v_linear = delta_pos / self.env.get_simulation_timestep()
        v_angular = delta_rot.as_euler('xyz', degrees=False) / self.env.get_simulation_timestep()
        return np.concatenate([v_linear, v_angular])


def get_random_initial_poses(n_demos):
    obj_bounding_box = [0.35, -0.4, 0.03, 0.85, 0.4, 0.03]
    obj_bounding_euler_angles = np.array([180, 0, 90, 180, 0, 270]) * np.pi/180
    obj_min = np.concatenate([obj_bounding_box[:3], obj_bounding_euler_angles[:3]]).reshape(1, 6)
    obj_max = np.concatenate([obj_bounding_box[3:], obj_bounding_euler_angles[3:]]).reshape(1, 6)
    obj_init_poses = np.random.uniform(obj_min, obj_max, (n_demos, 6))
    return obj_init_poses


def demo_reach_trajectory(env, arm, camera, target_obj, t_dummy, ref, max_steps, max_speed_linear, max_speed_angular, precision_linear, precision_angular, maintain, headless):
    """Collect demo images and actions for reaching a target dummy pose"""
    traj = Trajectory()

    controller = Controller(env, arm, t_dummy, ref)
    step_counter = 0
    error_counter = 0
    maintain_good_pose_counter = 0

    # Reach pose 5cm above target
    target1_position = np.array(target_obj.get_position(relative_to=ref)) + np.array([0, 0, 0.05])
    target1_quat = np.array(target_obj.get_quaternion(relative_to=ref))
    target1_pose = np.concatenate([target1_position, target1_quat])
    t_dummy.set_pose(target1_pose, relative_to=ref)

    dist_pos = controller.get_dist_linear()
    dist_angular = controller.get_dist_angular()

    while dist_pos > precision_linear or dist_angular > precision_angular or maintain_good_pose_counter < maintain:
        # Capture image and do action
        img = camera.capture_rgb()
        if dist_pos < precision_linear and dist_angular < precision_angular:
            maintain_good_pose_counter += 1
        else:
            maintain_good_pose_counter = 0
        try:
            v_tip = controller.step_with_displacement(max_speed_linear, max_speed_angular)
        except IKError as e:
            warnings.warn(str(e), Warning)
            error_counter += 1
            if error_counter < 10:
                env.step()
                continue
            raise e

        # Record the state action + other context info
        traj.add(img, v_tip)

        step_counter += 1
        dist_pos = controller.get_dist_linear()
        dist_angular = controller.get_dist_angular()

        if step_counter > max_steps:
            raise NotReachedError("Max steps per trajectory")


        if not headless:
            print('step', step_counter, end=' ')
            inp = input()
            if inp == 'q':
                env.shutdown()

    return traj


def collect_and_save_demos(env, arm, camera, target_obj, target_dummy, ref, n_demos, max_steps, save_demo_location, max_speed_linear, max_speed_angular, precision_linear, precision_angular, maintain, headless):
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
            traj = demo_reach_trajectory(env, arm, camera, target_obj, target_dummy, ref, max_steps, max_speed_linear, max_speed_angular, precision_linear, precision_angular, maintain, headless)
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

