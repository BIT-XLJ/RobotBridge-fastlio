import torch
import numpy as np
from envs.base_env import BaseEnv
from utils.motion_lib.motion_lib_robot import MotionLibRobot
from utils.motion_lib.rotations import get_euler_xyz_in_tensor, my_quat_rotate, calc_heading_quat_inv, quat_to_tan_norm, quat_mul
from loguru import logger
from utils.helpers import parse_observation

class MotionTracking(BaseEnv):
    def __init__(self, config):
        super().__init__(config)
        self.device = self.cfg.get('device', 'cpu')
        self._init_motion_lib()

    def _init_motion_lib(self):
        self.motion_cfg = self.cfg.motion
        self.motion_cfg.step_dt = self.simulator.high_dt
        self.smooth_end = self.motion_cfg.get("smooth_end", False)
        self._motion_lib = MotionLibRobot(self.motion_cfg, self.device)
        self.motion_ids = torch.arange(1).to(self.device)
        self.motion_start_times = np.zeros(1, dtype=np.float32)
        self.motion_start_idx = 0
        self.num_motions = self._motion_lib._num_unique_motions
        self.root_trans_offset = torch.zeros((1, 3),device=self.device)

    def _reset_envs(self, refresh):
        super()._reset_envs(refresh)
        logger.info(f'Aligning the Coordinate of Reference Motions and Current Robot State')
        logger.warning(f'Current Code has not supported assigning one reference motion to track; Check the data file and make sure it only contains one motion!')
        prop_obs_dict = self.simulator.update_obs()
        root_trans = torch.from_numpy(prop_obs_dict['root_trans']).to(self.device)
        root_quat = prop_obs_dict['root_quat']
        self._motion_lib.load_motions(target_heading=root_quat)
        self.motion_dt = self._motion_lib._motion_dt
        motion_times = torch.from_numpy(self.motion_start_times).to(self.device)
        motion_res = self._motion_lib.get_motion_state(self.motion_ids, motion_times)
        init_ref_root_trans = motion_res['root_pos']
        self.root_trans_offset = root_trans.unsqueeze(0) - init_ref_root_trans
        for i in range(20):
            self._update_obs()
        logger.info(f'Warmup Done!')

    def _physics_step(self):
        super()._physics_step()
        self.simulator.update_marker_pos(self.marker_pos.cpu().numpy())

    def _update_obs(self):
        super()._update_obs()
        motion_times = torch.from_numpy((self.episode_length_buf +1)*self.simulator.high_dt + self.motion_start_times).to(self.device)
        ref_motion_length = self._motion_lib.get_motion_length(self.motion_ids)
        self._ref_motion_phase = (motion_times/ref_motion_length).unsqueeze(0).cpu().numpy()

        motion_res = self._motion_lib.get_motion_state(self.motion_ids, motion_times, offset=self.root_trans_offset)
        ref_root_trans = motion_res["root_pos"]
        ref_body_pos_extend = self.marker_pos = motion_res["rg_pos_t"]
        ref_root_vel = motion_res["root_vel"]
        ref_root_rot = motion_res["root_rot"]
        ref_root_ang_vel = motion_res["root_ang_vel"]
        ref_joint_pos = motion_res["dof_pos"] # [num_envs, num_dofs]
        # print(f"knee dof position {ref_joint_pos[0, [3,9]]}, dof pos {self.dof_pos[0, [3,9]]}")

        heading_inv_rot = calc_heading_quat_inv(torch.from_numpy(self.root_quat).to(self.device), w_last=True)
        ref_heading_inv_rot = calc_heading_quat_inv(ref_root_rot.view(1,4), w_last=True)
        ref_heading_inv_rot_expand = ref_heading_inv_rot.unsqueeze(1).expand(-1, len(self.cfg.selected_keypoints_id), -1).reshape(-1, 4)

        # self._obs_ref_root_height = ref_body_pos_extend.view(1, -1 ,3)[:,0,2:]
        # ref_root_rot_rpy = get_euler_xyz_in_tensor(ref_root_rot.view(1,4))
        # roll = ref_root_rot_rpy[...,0]
        # pitch = ref_root_rot_rpy[...,1]
        # delta_yaw = ref_root_rot_rpy[..., 2] - torch.from_numpy(self.root_rpy[..., 2]).to(self.device)
        # self._obs_dif_global_root_rot_rpy = torch.stack([roll, pitch, delta_yaw], dim=-1)
        # self._obs_ref_local_root_vel = my_quat_rotate(ref_heading_inv_rot, ref_root_vel.view(1,3))
        # self._obs_ref_local_root_ang_vel = my_quat_rotate(ref_heading_inv_rot, ref_root_ang_vel.view(1,3))
        # self._obs_root_tracking_real = torch.cat([
        #     self._obs_ref_root_height,
        #     self._obs_ref_local_root_vel,
        #     self._obs_ref_local_root_ang_vel,
        # ], dim=-1).cpu().numpy()
        heading_inv_rot_expand = heading_inv_rot.unsqueeze(1).expand(-1, len(self.cfg.selected_keypoints_id), -1).reshape(-1, 4)
        ref_keypoints_rel_pos = (ref_body_pos_extend.view(1, -1, 3) - torch.from_numpy(self.root_trans).to(self.device))[:, self.cfg.selected_keypoints_id, :]
        self._obs_keypoint_tracking = my_quat_rotate(heading_inv_rot_expand, 
            ref_keypoints_rel_pos.view(-1, 3)).view(1, -1).cpu().numpy()    #_obs_keypoint_tracking 转换为了机身系下的相对位置

        pos_diff = ref_root_trans.view(1,3) - torch.from_numpy(self.root_trans).to(self.device)
        self._obs_local_ref_root_trans = my_quat_rotate(heading_inv_rot, pos_diff).view(1,-1)
        obs_local_ref_root_rot = quat_to_tan_norm(quat_mul(heading_inv_rot, ref_root_rot, w_last=True)).view(1, -1)
        self._obs_local_ref_root_rot = quat_to_tan_norm(quat_mul(heading_inv_rot, ref_root_rot, w_last=True)).view(1, -1)
        self._obs_local_ref_root_vel = my_quat_rotate(heading_inv_rot,ref_root_vel.view(1,3))
        self._obs_local_ref_root_ang_vel = my_quat_rotate(heading_inv_rot, ref_root_ang_vel.view(1,3))


        dif_local_root_lin_vel = self._obs_local_ref_root_vel - self.base_lin_vel
        self.diff_local_ang_vel = self._obs_local_ref_root_ang_vel - self.base_ang_vel
        diff_local_root_ang_vel_z = self.diff_local_ang_vel[:, 2:]
        self._obs_root_tracking = torch.cat([
            self._obs_local_ref_root_trans,                     # [num_envs, 3]
            self._obs_local_ref_root_rot,                       # [num_envs, 6]
            dif_local_root_lin_vel[:,:2],                       # [num_envs, 2] 
            diff_local_root_ang_vel_z                           # [num_envs, 1]
        ], dim=-1).cpu().numpy()
        obs_ref_local_lin_vel = my_quat_rotate(ref_heading_inv_rot, ref_root_vel)
        obs_ref_local_ang_vel = my_quat_rotate(ref_heading_inv_rot, ref_root_ang_vel)
        self._obs_root_tracking_real = torch.cat([
            obs_local_ref_root_rot,
            obs_ref_local_lin_vel[...,:2],
            obs_ref_local_ang_vel[...,2:]
        ], dim=-1).cpu().numpy()

        self.dif_joint_angles = ref_joint_pos - torch.from_numpy(self.dof_pos).to(self.device)
        self._obs_joint_tracking = self.dif_joint_angles.view(1, -1).cpu().numpy()
    
    def _get_obs_ref_motion_phase(self):
        return self._ref_motion_phase
    
    def _get_obs_root_tracking(self):
        return self._obs_root_tracking
    
    def _get_obs_root_tracking_real(self):
        return self._obs_root_tracking_real
    
    def _get_obs_keypoint_tracking(self):
        return self._obs_keypoint_tracking
    
    def _get_obs_joint_tracking(self):
        return self._obs_joint_tracking
    
    def _get_obs_episodic_obs(self):
        # this will return the episodic observations
        # obs can be reshape num_envs * frame * obs_dim
        assert 'episodic_obs' in self.cfg.obs.obs_auxiliary.keys()
        episodic_config = self.cfg.obs.obs_auxiliary['episodic_obs']
        # assert len is same in episodic_config
        frames = list(episodic_config.values())
        assert len(set(frames)) == 1, "All episodic obs should have the same length"
        num_frames = frames[0]

        obs_list = episodic_config.keys()
        current_obs = {}
        parse_observation(self, obs_list, current_obs, self.cfg.obs.obs_scales)
        episodic_obs = []
        for key in sorted(obs_list):            
            history_tensor = self.history_handler.query(key)[:num_frames-1] # note the order of the history
            history_tensor = np.concatenate([current_obs[key], history_tensor], axis=0)  #(history_frame, obs_dim)
            # num_envs, num_frames, obs_dim
            episodic_obs.append(history_tensor.reshape(num_frames, -1)[::-1])
        episodic_obs = np.concatenate(episodic_obs, axis=1)
        return episodic_obs.reshape(1,-1)
    
    def _check_termination(self):
        hard_reset = self.simulator.check_termination()  # failed
        soft_reset = not self.smooth_end and (self._ref_motion_phase >= 1.0) # success

        if not hard_reset and soft_reset:  # Not terminated but the task is done!
            self._save_collected_traj(True)
            self._reset_envs(False)
            self.compute_observation()
        elif hard_reset:
            self._save_collected_traj(False)
            self._reset_envs(True)
            self.compute_observation()
            