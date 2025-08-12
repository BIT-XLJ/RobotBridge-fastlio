import numpy as np

def parse_observation(cls, key_list, buf_dict, obs_scales):
    for obs_key in key_list:
        actor_obs = getattr(cls, f'_get_obs_{obs_key}')().copy()
        obs_scale = np.array(obs_scales[obs_key], dtype=np.float32)
        buf_dict[obs_key] = actor_obs*obs_scale

def get_gravity(quat,w_last=True):
    if w_last:
        qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
    else:
        qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]

    gravity = np.zeros(3)
    gravity[0] = 2*(-qz*qx + qw*qy)
    gravity[1] = -2*(qz*qy + qw*qx)
    gravity[2] = 1 - 2*(qw*qw + qz*qz)

    return gravity

def get_rpy(quat, w_last=True):
    if w_last:
        qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
    else:
        qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
    
    sinr_cosp = 2.0 * (qw*qx+qy*qz)
    cosr_cosp = qw*qw - qx*qx - qy*qy +qz*qz
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2*(qw*qy - qz*qx)
    pitch = np.where(
        np.abs(sinp)>=1, np.abs(np.pi/2)*np.sign(sinp), np.arcsin(sinp)
    )
    
    siny_cosp = 2.0 * (qw*qz + qx*qy)
    cosy_cosp = qw*qw + qx*qx - qy*qy -qz*qz
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.stack([roll, pitch, yaw])