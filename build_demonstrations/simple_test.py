import numpy as np
from pyrep.errors import IKError
import warnings
import torch


def set_controls(arm, action, ref, counter_error):
    current_pos = arm.get_tip().get_position(relative_to=ref)
    next_pos = current_pos + action
    next_quat = arm.get_tip().get_quaternion(relative_to=ref)
    q = np.array(arm.get_joint_positions())
    try:
        q_next = arm.solve_ik_via_jacobian(next_pos, quaternion=next_quat, relative_to=ref)
    except IKError as e:
        arm.set_joint_target_velocities([0] * arm.get_joint_count())
        counter_error += 1
        env.step()
        return
    v = (q_next - q) / env.get_simulation_timestep()
    arm.set_joint_target_velocities(v)
    return counter_error

    
def format_input(x):
    return torch.tensor(x.flatten(), dtype=torch.float32)[None]


def format_output(x, mean, std):
    return x.detach().numpy().flatten() * std + mean
    

def test_model(model, env, camera, objects, arm, ref, mean, std, max_steps):
    arm.set_control_loop_enabled(False)
    print("Starting demo")
    env.stop()
    env.start()
    counter_error = 0
    for step in range(max_steps):
        print(f"step #{step}/{max_steps}")
        img_input = camera.capture_rgb()
        env.step() # Important to get the image
        with torch.no_grad():
            x = format_input(img_input)
            y = model.forward(x)
            act_output = format_output(y, mean, std)
        counter_error = set_controls(arm, act_outp, ref, counter_error)
        if counter_error >= 10:
            warnings.warn('10x IKError', Warning)
            break
        x = input()
        if x=='b':
            break
        if x=='q':
            exit()
    env.stop()
    env.shutdown()
