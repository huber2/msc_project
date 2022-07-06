import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp

class Controller:
    def __init__(self, env, arm, t_dummy, reference_frame):
        self.env = env
        self.arm = arm
        self.t_dummy = t_dummy
        self.ref = reference_frame
        
    def get_pose_step(self, max_speed_linear, max_speed_angular):
        
        # Position error
        tip_pos = np.array(self.arm.get_tip().get_position(relative_to=self.ref))
        target_pos = np.array(self.t_dummy.get_position(relative_to=self.ref))
        diff_pos = target_pos - tip_pos
        dist_pos = np.linalg.norm(diff_pos)
        

        # Orientation error
        tip_quat = self.arm.get_tip().get_quaternion(relative_to=self.ref)
        target_quat = self.t_dummy.get_quaternion(relative_to=self.ref)
        
        tip_rot = Rotation.from_quat(tip_quat)
        target_rot = Rotation.from_quat(target_quat)
        
        diff_rot = target_rot * tip_rot.inv()
        dist_angle = diff_rot.magnitude()
        
        # time step
        dt = self.env.get_simulation_timestep()
        
        # Position
        step_speed_pos = dist_pos / dt
        coef_speed_pos = min(1, max_speed_linear/step_speed_pos)
        delta_pos = diff_pos * coef_speed_pos
        next_pos = tip_pos + delta_pos
        
        # Orientation
        step_speed_ori = dist_angle / dt
        coef_ori_step = min(1, max_speed_angular/step_speed_ori)
        
        key_rotations = Rotation.concatenate([tip_rot, target_rot])
        key_times = [0, 1]
        slerp = Slerp(key_times, key_rotations)
        next_rot = slerp(coef_ori_step)
        next_quat = next_rot.as_quat()

        delta_rot = next_rot * tip_rot.inv()
        delta_quat = delta_rot.as_quat()
        
        return delta_pos, delta_quat, next_pos, next_quat
    
    def demo_step(self, next_pos, next_quat):
        """Compute the joint angular velocities with inverse kinematics"""
        q_next = self.arm.solve_ik_via_jacobian(next_pos, quaternion=next_quat, relative_to=self.ref)
        q_current = np.array(self.arm.get_joint_positions())
        v = (q_next - q_current) / self.env.get_simulation_timestep()
        self.arm.set_joint_target_velocities(v)
        self.env.step()
        return q_current, v

    def get_distances(self):
        # Position error
        tip_pos = np.array(self.arm.get_tip().get_position(relative_to=self.ref))
        target_pos = np.array(self.t_dummy.get_position(relative_to=self.ref))
        diff_pos = target_pos - tip_pos
        dist_pos = np.linalg.norm(diff_pos)
        

        # Orientation error
        tip_quat = self.arm.get_tip().get_quaternion(relative_to=self.ref)
        target_quat = self.t_dummy.get_quaternion(relative_to=self.ref)
        tip_rot = Rotation.from_quat(tip_quat)
        target_rot = Rotation.from_quat(target_quat)
        diff_rot = target_rot * tip_rot.inv()
        dist_angle = diff_rot.magnitude()

        return dist_pos, dist_angle
    
    
    def get_velocity_step(self, max_speed_linear, max_speed_angular):
        
        # Position error
        tip_pos = np.array(self.arm.get_tip().get_position(relative_to=self.ref))
        target_pos = np.array(self.t_dummy.get_position(relative_to=self.ref))
        diff_pos = target_pos - tip_pos
        dist_pos = np.linalg.norm(diff_pos)
        

        # Orientation error
        tip_quat = self.arm.get_tip().get_quaternion(relative_to=self.ref)
        target_quat = self.t_dummy.get_quaternion(relative_to=self.ref)
        
        tip_rot = Rotation.from_quat(tip_quat)
        target_rot = Rotation.from_quat(target_quat)
        
        diff_rot = target_rot * tip_rot.inv()
        dist_angle = diff_rot.magnitude()
        
        #
        dt = self.env.get_simulation_timestep()
        
        # Position
        step_speed_pos = dist_pos / dt
        coef_speed_pos = min(1, max_speed_linear/step_speed_pos)
        delta_pos = diff_pos * coef_speed_pos
        next_pos = tip_pos + delta_pos
        
        # Orientation
        step_speed_ori = dist_angle / dt
        coef_ori_step = min(1, max_speed_angular/step_speed_ori)
        
        key_rotations = Rotation.concatenate([tip_rot, target_rot])
        key_times = [0, 1]
        slerp = Slerp(key_times, key_rotations)
        next_rot = slerp(coef_ori_step)
        next_quat = next_rot.as_quat()

        delta_rot = next_rot * tip_rot.inv()
        delta_quat = delta_rot.as_quat()
        
        return delta_pos, delta_quat, next_pos, next_quat
    
    
    def step_with_jac(self, delta_pos, delta_quat):
        delta_tip = np.concatenate([delta_pos, delta_quat]).reshape(-1, 1)
        jac = arm.get_jacobian()
        delta_q = np.linalg.pinv(jac) @ delta_tip
        dt = self.env.get_simulation_timestep()
        arm.set_joint_target_velocities(delta_q / dt)
        env.step()
        
        
    def get_velocity_step(self, max_speed_linear, max_speed_angular):
        
        # Position error
        tip_pos = np.array(self.arm.get_tip().get_position(relative_to=self.ref))
        target_pos = np.array(self.t_dummy.get_position(relative_to=self.ref))
        diff_pos = target_pos - tip_pos
        dist_pos = np.linalg.norm(diff_pos)
        

        # Orientation error
        tip_quat = self.arm.get_tip().get_quaternion(relative_to=self.ref)
        target_quat = self.t_dummy.get_quaternion(relative_to=self.ref)
        
        tip_rot = Rotation.from_quat(tip_quat)
        target_rot = Rotation.from_quat(target_quat)
        
        diff_rot = target_rot * tip_rot.inv()
        dist_angle = diff_rot.magnitude()
        
        # time step
        dt = self.env.get_simulation_timestep()
        
        # Position
        coef_speed_pos = min(1, max_speed_linear*dt/dist_pos)
        v_linear = diff_pos * coef_speed_pos / dt
        
        # Orientation
        coef_ori_step = min(1, max_speed_angular*dt/dist_angle)
        key_rotations = Rotation.concatenate([tip_rot, target_rot])
        key_times = [0, 1]
        slerp = Slerp(key_times, key_rotations)
        next_rot = slerp(coef_ori_step)
        delta_rot = next_rot * tip_rot.inv()
        delta_euler = delta_rot.as_euler('zyx') # extrinsic
        v_angular_zyx = delta_euler / dt
        
        return v_linear, v_angular_zyx
        
    
