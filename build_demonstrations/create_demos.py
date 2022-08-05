import numpy as np
import warnings
from pyrep.errors import IKError
from demos_control_new import Controller


class NotReachedError(Exception):
    """When the robot fails to reach a target"""


class Trajectory:
    """Store sequence of images (observations) and actions"""
    def __init__(self):
        self.image_seq = []
        self.action_seq = []

    def __len__(self):
        return len(self.action_seq)

    def add(self, image, action):
        """
        Args:
            image (array): image with rgb values between 0 and 1
            action (array): linear velocity [v_x, v_y, v_z] (m/s)
        """
        image_8bit = np.array(image * 255, dtype=np.uint8)
        self.image_seq.append(image_8bit)
        self.action_seq.append(action)


class Demonstrations:
    """Store demonstration trajectories. The sequences of images and actions collected are
    concatenated to be saved as numpy npz format. Since the sequences can be of diffferent 
    lengths a step marker is used to indicate the start/finish of each sequence.
    """
    def __init__(self):
        self.step_marker = [0]
        self.demo_image_sequences = []
        self.demo_action_sequences = []

    def add_trajectory(self, trajectory):
        self.step_marker.append(self.step_marker[-1] + len(trajectory))
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
         
    def get_distance(self):
        """Distance (in m) between the robot's end effector and the target"""
        return self.arm.get_tip().check_distance(self.tgt)
    
    def get_tip_linear_displacement(self, max_speed_linear):
        """Linear dispacement to find next position towards target"""
        tgt_pos = np.array(self.tgt.get_position(relative_to=self.ref))
        tip_pos = np.array(self.arm.get_tip().get_position(relative_to=self.ref))
        diff_pos = tgt_pos - tip_pos
        dist_pos = np.linalg.norm(diff_pos)
        whole_step_speed_linear = dist_pos / self.env.get_simulation_timestep()
        coef_step_linear = min(1, max_speed_linear/whole_step_speed_linear)
        delta_pos = diff_pos * coef_step_linear
        next_pos = tip_pos + delta_pos
        return delta_pos, next_pos

    def get_joint_velocities_with_ik_solver(self, next_tip_pos, next_tip_quat):
        """Compute robot joint velocities with inverse kinematics using the Jacobian method in CoppeliaSim"""
        next_joint_angles = self.arm.solve_ik_via_jacobian(next_tip_pos, quaternion=next_tip_quat, relative_to=self.ref)
        current_joint_angles = np.array(self.arm.get_joint_positions())
        v_joints = (next_joint_angles - current_joint_angles) / self.env.get_simulation_timestep()
        return v_joints
    
    def step_linear_velocity(self, max_speed_linear):
        """Step towards target and return the corresponding linear velocity"""
        delta_pos, next_pos = self.get_tip_linear_displacement(max_speed_linear)
        next_quat = self.tgt.get_quaternion(relative_to=self.ref)
        v_joints = self.get_joint_velocities_with_ik_solver(next_pos, next_quat)
        self.arm.set_joint_target_velocities(v_joints)
        self.env.step()
        v_linear = delta_pos / self.env.get_simulation_timestep()
        return v_linear


def demo_reach_trajectory(env, arm, camera, tgt_obj, tgt_dummy, ref, max_steps, max_speed_linear, precision_linear, maintain, headless):
    """Collect demo images and actions for reaching a target dummy pose.
    The target dummy pose is set 5cm above the target object with the same orientation as the robot's end effector.
    The robot end effector has to reach the target dummy and stay within a close range during several time steps
    in order to complete the demonstartion sucessfully."""
    traj = Trajectory()

    controller = Controller(env, arm, tgt_dummy, ref)
    step_counter = 0
    error_counter = 0
    maintain_good_pose_counter = 0

    # Reach dummy pose 5cm above target object
    target1_position = np.array(tgt_obj.get_position(relative_to=ref)) + np.array([0, 0, 0.05])
    target1_quat = np.array(arm.get_tip().get_quaternion(relative_to=ref))
    target1_pose = np.concatenate([target1_position, target1_quat])
    tgt_dummy.set_pose(target1_pose, relative_to=ref)

    dist_pos = controller.get_distance()

    while dist_pos > precision_linear or maintain_good_pose_counter < maintain:
        step_counter += 1
        if step_counter > max_steps:
            raise NotReachedError("Max steps per trajectory")
        if dist_pos < precision_linear:
            maintain_good_pose_counter += 1
        else:
            maintain_good_pose_counter = 0

        # Capture image and do action
        img = camera.capture_rgb()
        try:
            v_linear = controller.step_linear_velocity(max_speed_linear)
        except IKError as e:
            warnings.warn(str(e), Warning)
            error_counter += 1
            if error_counter < 10:
                env.step()
                continue
            raise e

        # Record the state and action
        traj.add(img, v_linear)
        dist_pos = controller.get_distance()

        if not headless:
            # Wait for human to play next action
            print('step', step_counter, end=' ')
            inp = input()
            if inp == 'q':
                env.shutdown()
    return traj


def get_random_tgt_pose(bounding_box, bounding_angles):
    pose_min = np.concatenate([bounding_box[:3], bounding_angles[:3]])
    pose_max = np.concatenate([bounding_box[3:], bounding_angles[3:]])
    pose = np.random.uniform(pose_min, pose_max, 6)
    return pose


def collect_and_save_demos(env, arm, camera, tgt_obj, tgt_dummy, ref, tgt_bounding_box, tgt_bounding_angles, n_demos, max_steps, max_speed_linear, precision_linear, maintain, headless, save_demo_location):
    """Collect and save a set of demonstration trajectories in numpy npz format

    Args:
        env: pyrep environment
        arm: pyrep robotic arm
        tgt_obj: pyrep target object/shape
        tgt_dummy: pyrep dummy for the target pose
        ref: pyrep dummy for the reference frame
        tgt_bounding_box: axis-aligned bounding box for target position
            [min_x, min_y, min_z, max_x, max_y, max_z] (in m)
        tgt_bounding_angles: bounding euler angles for target orientation (xyz extrinsic)
            [min_alpha, min_beta, min_gamma, max_alpha, max_beta, max_gamma] (in rad)
        n_demos (int): number of demonstration to collect
        max_steps (int): maximum number of time steps per demonstration
        max_speed_linear (float): maximum movement speed (m/s) of the robot's end effector
        precision_linear (float): distance from the target dummy defining the target zone
        maintain (int): number of time steps the end effector has to stay in the target zone
        headless (bool): run simulation without Coppelia display. If False press enter at each time step
        save_demo_location (str): path to save the npz file with the demos data

    """
    arm.set_control_loop_enabled(False)
    counter_demo_done = 0
    counter_tries = 0
    demos = Demonstrations()

    while counter_demo_done < n_demos:
        env.stop()
        counter_tries += 1
        print(f"Try #{counter_tries}")
        tgt_pose = get_random_tgt_pose(tgt_bounding_box, tgt_bounding_angles)
        env.start()
        tgt_obj.set_position(tgt_pose[:3], relative_to=ref)
        tgt_obj.set_orientation(tgt_pose[3:], relative_to=ref)
        try:
            traj = demo_reach_trajectory(env, arm, camera, tgt_obj, tgt_dummy, ref, max_steps, max_speed_linear, precision_linear, maintain, headless)
        except (NotReachedError, IKError) as e:
            warnings.warn(str(e), Warning)
            continue
        env.stop()
        demos.add_trajectory(traj)
        counter_demo_done += 1
        print(f"Demos collected {counter_demo_done}/{n_demos}")
        if counter_demo_done%100 == 0:
            demos.save(save_demo_location)
            print(f'Collected and saved {counter_demo_done} demos!')
    env.shutdown()
    demos.save(save_demo_location)
    print(f"Total: {counter_demo_done} demos successfully collected!")

