import numpy as np
from pyrep.errors import IKError
import warnings
import torch


def set_controls(env, arm, action, ref, counter_error, init_quat):
    current_pos = arm.get_tip().get_position(relative_to=ref)
    next_pos = current_pos + action
    next_quat = init_quat
    q = np.array(arm.get_joint_positions())
    try:
        q_next = arm.solve_ik_via_jacobian(next_pos, quaternion=next_quat, relative_to=ref)
    except IKError as e:
        arm.set_joint_target_velocities([0] * arm.get_joint_count())
        counter_error += 1
        return counter_error
    v = (q_next - q) / env.get_simulation_timestep()
    arm.set_joint_target_velocities(v)
    return counter_error

    

def test_model(model, env, camera, arm, ref, max_steps, format_input, format_output):
    arm.set_control_loop_enabled(False)
    env.stop()
    env.start()
    init_quat = arm.get_tip().get_quaternion(relative_to=ref)
    counter_error = 0
    for step in range(max_steps):
        print(f"step #{step}/{max_steps}", end=" ")
        img_input = camera.capture_rgb()
        env.step() # Important to get the image
        with torch.no_grad():
            x = format_input(img_input)
            y = model.forward(x)
            act_output = format_output(y)
        counter_error = set_controls(env, arm, act_output, ref, counter_error, init_quat)

        if counter_error >= 10:
            warnings.warn('10x IKError', Warning)
            break
        x = input()
        if x=='b':
            break
        if x=='q':
            exit()
    env.stop()
