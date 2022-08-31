import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp

class Controller:

    def __init__(self, env, arm, target_dummy, reference_dummy):
        self.env = env
        self.arm = arm
        self.tgt = target_dummy
        self.ref = reference_dummy

    def get_joint_velocities_from_tip_velocity_with_jacobian(self, v_tip):
        jac = self.arm.get_jacobian()
        v_joints = np.linalg.pinv(jac) @ v_tip
        return v_joints

    def get_joint_velocities_from_tip_displacement_with_ik_solver(self, next_tip_pos, next_tip_quat):
        next_joint_angles = self.arm.solve_ik_via_jacobian(next_tip_pos, quaternion=next_tip_quat, relative_to=self.ref)
        current_joint_angles = np.array(self.arm.get_joint_positions())
        v_joints = (next_joint_angles - current_joint_angles) / self.env.get_simulation_timestep()
        return v_joints
    
    def get_tip_linear_velocity(self, max_speed_linear):
        # Position error
        diff_pos = np.array(self.tgt.get_position(relative_to=self.arm.get_tip()))
        dist_pos = np.linalg.norm(diff_pos)

        # time step
        dt = self.env.get_simulation_timestep()
        
        # Linear velocity
        whole_step_speed_linear = dist_pos / dt
        coef_step_linear = min(1, max_speed_linear/whole_step_speed_linear)
        v_linear = coef_step_linear * diff_pos / dt
        return v_linear

    def get_tip_angular_velocity(self, max_speed_angular):
        # Orientation error
        tip_quat = self.arm.get_tip().get_quaternion(relative_to=self.ref)
        target_quat = self.tgt.get_quaternion(relative_to=self.ref)
        
        tip_rot = Rotation.from_quat(tip_quat)
        target_rot = Rotation.from_quat(target_quat)
        diff_rot = target_rot * tip_rot.inv()
        dist_angle = diff_rot.magnitude()

        # time step
        dt = self.env.get_simulation_timestep()

        # Angular velocity
        whole_step_speed_angular = dist_angle / dt
        coef_step_angular = min(1, max_speed_angular/whole_step_speed_angular)

        key_rotations = Rotation.concatenate([tip_rot, target_rot])
        key_times = [0, dt]
        slerp = Slerp(key_times, key_rotations)
        rot_after_1sec = slerp(coef_step_angular)
        rot_diff_after_1sec = rot_after_1sec * tip_rot.inv()
        v_angular = rot_diff_after_1sec.as_euler('zyx') # extrinsic
        return v_angular

    def get_tip_linear_displacement(self, max_speed_linear):
        # Position error
        diff_pos = np.array(self.tgt.get_position(relative_to=self.arm.get_tip()))
        dist_pos = np.linalg.norm(diff_pos)

        # time step
        dt = self.env.get_simulation_timestep()
        
        # Linear displacement
        whole_step_speed_linear = dist_pos / dt
        coef_step_linear = min(1, max_speed_linear/whole_step_speed_linear)
        delta_pos = diff_pos * coef_step_linear
        next_pos = np.array(self.arm.get_tip().get_position()) + delta_pos
        return delta_pos, next_pos

    def get_tip_angular_displacement(self, max_speed_angular):
        # Orientation error
        tip_quat = self.arm.get_tip().get_quaternion(relative_to=self.ref)
        target_quat = self.tgt.get_quaternion(relative_to=self.ref)
        
        tip_rot = Rotation.from_quat(tip_quat)
        target_rot = Rotation.from_quat(target_quat)
        diff_rot = target_rot * tip_rot.inv()
        dist_angle = diff_rot.magnitude()

        # time step
        dt = self.env.get_simulation_timestep()

        # Angular displacement
        whole_step_speed_angular = dist_angle / dt
        coef_step_angular = min(1, max_speed_angular/whole_step_speed_angular)
        
        key_rotations = Rotation.concatenate([tip_rot, target_rot])
        key_times = [0, 1]
        slerp = Slerp(key_times, key_rotations)
        next_rot = slerp(coef_step_angular)
        next_quat = next_rot.as_quat()

        delta_rot = next_rot * tip_rot.inv()
        delta_quat = delta_rot.as_quat()
        return delta_quat, next_quat

    def step_with_velocities(self, max_speed_linear, max_speed_angular):
        v_linear = self.get_tip_linear_velocity(max_speed_linear)
        v_angular = self.get_tip_angular_velocity(max_speed_angular)
        v_tip = np.concatenate([v_linear, v_angular]).reshape(-1, 1)
        v_joints = self.get_joint_velocities_from_tip_velocity_with_jacobian(v_tip)
        self.arm.set_joint_target_velocities(v_joints)
        self.env.step()
        return v_linear, v_angular

    def step_with_displacement(self, max_speed_linear, max_speed_angular):
        delta_pos, next_pos = self.get_tip_linear_displacement(max_speed_linear)
        delta_quat, next_quat = self.get_tip_angular_displacement(max_speed_angular)
        v_joints = self.get_joint_velocities_from_tip_displacement_with_ik_solver(next_pos, next_quat)
        self.arm.set_joint_target_velocities(v_joints)
        self.env.step()
        return delta_pos, delta_quat

    def get_distance_linear(self):
        # Position error
        diff_pos = np.array(self.tgt.get_position(relative_to=self.arm.get_tip()))
        dist_pos = np.linalg.norm(diff_pos)
        return dist_pos

    def get_distance_angular(self):
        # Orientation error
        tip_quat = self.arm.get_tip().get_quaternion(relative_to=self.ref)
        target_quat = self.tgt.get_quaternion(relative_to=self.ref)
        tip_rot = Rotation.from_quat(tip_quat)
        target_rot = Rotation.from_quat(target_quat)
        diff_rot = target_rot * tip_rot.inv()
        dist_angle = diff_rot.magnitude()
        return dist_angle
        
    
