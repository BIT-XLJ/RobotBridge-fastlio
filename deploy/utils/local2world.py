import pytorch_kinematics as pk
import numpy as np
import torch 

_l2w_robot_file = "deploy/data/assets/g1/g1_29dof_anneal_23dof.urdf"
_l2w_joint_name = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', 'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint']
_l2w_body_name = ['pelvis',
 'left_hip_pitch_link',
 'left_hip_roll_link',
 'left_hip_yaw_link',
 'left_knee_link',
 'left_ankle_pitch_link',
 'left_ankle_roll_link',
 'right_hip_pitch_link',
 'right_hip_roll_link',
 'right_hip_yaw_link',
 'right_knee_link',
 'right_ankle_pitch_link',
 'right_ankle_roll_link',
 'waist_yaw_link',
 'waist_roll_link',
 'torso_link',
 'left_shoulder_pitch_link',
 'left_shoulder_roll_link',
 'left_shoulder_yaw_link',
 'left_elbow_link',
 'left_wrist_roll_link',
 'left_wrist_pitch_link',
 'left_wrist_yaw_link',
 'right_shoulder_pitch_link',
 'right_shoulder_roll_link',
 'right_shoulder_yaw_link',
 'right_elbow_link',
 'right_wrist_roll_link',
 'right_wrist_pitch_link',
 'right_wrist_yaw_link'] 
_l2w_chain = pk.build_chain_from_urdf(open(_l2w_robot_file).read())

def fk_dof(dof_pos):
    # d = "cuda" if torch.cuda.is_available() else "cpu"
    # dtype = torch.float64
    # chain = chain.to(dtype=dtype, device=d)


    dof_dic = {}
    for i, jn in enumerate(_l2w_joint_name):
        dof_dic[jn] = dof_pos[i]
    ret = _l2w_chain.forward_kinematics(dof_dic)
    # recall that we specify joint values and get link transforms
    all_joint_global_pose = torch.zeros((30, 3))
    all_joint_global_quan = torch.zeros((30, 4))

    for i in range(len(_l2w_body_name)):
        # import pdb; pdb.set_trace()
        # tg = ret[body_name[i].replace('_link', '')]
        tg = ret[_l2w_body_name[i]]
        m = tg.get_matrix()
        pos = m[:, :3, 3]
        rot = pk.matrix_to_quaternion(m[:, :3, :3])
        all_joint_global_pose[i] = pos
        all_joint_global_quan[i] = rot
    return all_joint_global_pose, all_joint_global_quan