import glob
import os.path as osp
import numpy as np
import joblib
import torch
from scipy.spatial.transform import Rotation as sRot

from enum import Enum
from utils.motion_lib.skeleton import SkeletonTree
from pathlib import Path
from easydict import EasyDict
from loguru import logger
from rich.progress import track

from utils.motion_lib.rotations import slerp, calc_heading_quat_inv

class FixHeightMode(Enum):
    no_fix = 0
    full_fix = 1
    ankle_fix = 2

class MotionlibMode(Enum):
    file = 1
    directory = 2


def to_torch(tensor):
    if torch.is_tensor(tensor):
        return tensor
    else:
        return torch.from_numpy(tensor)

class MotionLibBase():
    def __init__(self, motion_lib_cfg, num_envs, device):
        self.m_cfg = motion_lib_cfg
        self._sim_fps = 1/self.m_cfg.get("step_dt", 1/50)
        
        self.num_envs = num_envs
        self._device = device
        self.mesh_parsers = None
        self.has_action = False
        skeleton_file = Path(self.m_cfg.asset.asset_root) / self.m_cfg.asset.asset_file
        self.skeleton_tree = SkeletonTree.from_mjcf(skeleton_file)
        logger.info(f"Loaded skeleton from {skeleton_file}")
        logger.info(f"Loading motion data from {self.m_cfg.motion_file}...")
        self.load_data(self.m_cfg.motion_file)
        self.setup_constants(fix_height = False,  multi_thread = False)
        self.align_method = self.m_cfg.get('align_method', None)
        self.smooth_init = self.m_cfg.get('smooth_init', False)
        self.smooth_init_frames = self.m_cfg.get('smooth_init_frames', 30)
        self.smooth_end = self.m_cfg.get('smooth_end', False)
        self.smooth_end_frames = self.m_cfg.get('smooth_end_frames', 30)
        self.default_angles = self.m_cfg.get('default_angles', [])
        return
        
    def load_data(self, motion_file, min_length=-1, im_eval = False):
        if osp.isfile(motion_file):
            self.mode = MotionlibMode.file
            self._motion_data_load = joblib.load(motion_file)
        else:
            self.mode = MotionlibMode.directory
            self._motion_data_load = glob.glob(osp.join(motion_file, "*.pkl"))
        data_list = self._motion_data_load
        if self.mode == MotionlibMode.file:
            if min_length != -1:
                # filtering the data by the length of the motion
                data_list = {k: v for k, v in list(self._motion_data_load.items()) if len(v['pose_quat_global']) >= min_length}
            elif im_eval:
                # sorting the data by the length of the motion
                data_list = {item[0]: item[1] for item in sorted(self._motion_data_load.items(), key=lambda entry: len(entry[1]['pose_quat_global']), reverse=True)}
            else:
                data_list = self._motion_data_load
            self._motion_data_list = np.array(list(data_list.values()))
            self._motion_data_keys = np.array(list(data_list.keys()))
        else:
            self._motion_data_list = np.array(self._motion_data_load)
            self._motion_data_keys = np.array(self._motion_data_load)
        
        self._num_unique_motions = len(self._motion_data_list)
        if self.mode == MotionlibMode.directory:
            self._motion_data_load = joblib.load(self._motion_data_load[0]) # set self._motion_data_load to a sample of the data 
        logger.info(f"Loaded {self._num_unique_motions} motions")

    def setup_constants(self, fix_height = FixHeightMode.full_fix, multi_thread = True):
        self.fix_height = fix_height
        self.multi_thread = multi_thread
        
        #### Termination history
        self._curr_motion_ids = None
        self._sampling_prob = torch.ones(self._num_unique_motions).to(self._device) / self._num_unique_motions  # For use in sampling batches

    def get_motion_state(self, motion_ids, motion_times, offset=None):
        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        local_rot0 = self.dof_pos[f0l]
        local_rot1 = self.dof_pos[f1l]
            
        body_vel0 = self.gvs[f0l]
        body_vel1 = self.gvs[f1l]

        body_ang_vel0 = self.gavs[f0l]
        body_ang_vel1 = self.gavs[f1l]

        rg_pos0 = self.gts[f0l, :]
        rg_pos1 = self.gts[f1l, :]

        dof_vel0 = self.dvs[f0l]
        dof_vel1 = self.dvs[f1l]

        vals = [local_rot0, local_rot1, body_vel0, body_vel1, body_ang_vel0, body_ang_vel1, rg_pos0, rg_pos1, dof_vel0, dof_vel1]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)

        blend_exp = blend.unsqueeze(-1)

        if offset is None:
            rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1  # ZL: apply offset
        else:
            rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1 + offset[..., None, :]  # ZL: apply offset

        body_vel = (1.0 - blend_exp) * body_vel0 + blend_exp * body_vel1
        body_ang_vel = (1.0 - blend_exp) * body_ang_vel0 + blend_exp * body_ang_vel1

        dof_pos = (1.0 - blend) * local_rot0 + blend * local_rot1

        rb_rot0 = self.grs[f0l]
        rb_rot1 = self.grs[f1l]
        rb_rot = slerp(rb_rot0, rb_rot1, blend_exp)
        return_dict = {}
        
        if "gts_t" in self.__dict__:
            rg_pos_t0 = self.gts_t[f0l]
            rg_pos_t1 = self.gts_t[f1l]
            if offset is None:
                rg_pos_t = (1.0 - blend_exp) * rg_pos_t0 + blend_exp * rg_pos_t1  
            else:
                rg_pos_t = (1.0 - blend_exp) * rg_pos_t0 + blend_exp * rg_pos_t1 + offset[..., None, :]
        else:
            rg_pos_t = rg_pos

        return_dict.update({
            "dof_pos": dof_pos.clone(),
            "root_ang_vel": body_ang_vel[..., 0, :].clone(),
            "root_vel": body_vel[..., 0, :].clone(),
            "root_rot": rb_rot[..., 0, :].clone(),
            "root_pos": rg_pos[..., 0, :].clone(),
            "rg_pos_t": rg_pos_t,
        })
        return return_dict
    
    def load_motions(self, 
                     start_idx=0, 
                     target_heading = None,
                     random_sample=True):
        # import ipdb; ipdb.set_trace()

        motions = []
        _motion_lengths = []
        _motion_fps = []
        _motion_dt = []
        _motion_num_frames = []
        _motion_bodies = []
        _motion_aa = []

        total_len = 0.0
        self.num_joints = len(self.skeleton_tree.node_names)
        num_motion_to_load = self.num_envs

        # sample_idxes = torch.remainder(torch.arange(num_motion_to_load) + start_idx, self._num_unique_motions ).to(self._device)

        if random_sample:

            sample_idxes = torch.multinomial(self._sampling_prob, num_samples=num_motion_to_load, replacement=True).to(self._device)

        else:

            sample_idxes = torch.remainder(torch.arange(num_motion_to_load) + start_idx, self._num_unique_motions ).to(self._device)
        self._curr_motion_ids = sample_idxes
        self.curr_motion_keys = self._motion_data_keys[sample_idxes.cpu()]

        logger.info(f"Loading {num_motion_to_load} motions...")
        logger.info(f"Sampling motion: {sample_idxes[:5]}, ....")
        logger.info(f"Current motion keys: {self.curr_motion_keys}, ....")

        motion_data_list = self._motion_data_list[sample_idxes.cpu().numpy()]
        res_acc = self.load_motion_with_skeleton(motion_data_list, target_heading)
        for f in track(range(len(res_acc)), description="Loading motions..."):
            motion_file_data, curr_motion = res_acc[f]
            motion_fps = curr_motion.fps
            curr_dt = 1.0 / motion_fps
            num_frames = curr_motion.global_rotation.shape[0]
            curr_len = 1.0 / motion_fps * (num_frames - 1)

            if "beta" in motion_file_data:
                _motion_aa.append(motion_file_data['pose_aa'].reshape(-1, self.num_joints * 3))
                _motion_bodies.append(curr_motion.gender_beta)
            else:
                _motion_aa.append(np.zeros((num_frames, self.num_joints * 3)))
                _motion_bodies.append(torch.zeros(17))

            _motion_fps.append(motion_fps)
            _motion_dt.append(curr_dt)
            _motion_num_frames.append(num_frames)
            motions.append(curr_motion)
            _motion_lengths.append(curr_len)
            del curr_motion
        
        self._motion_lengths = torch.tensor(_motion_lengths, device=self._device, dtype=torch.float32)
        self._motion_fps = torch.tensor(_motion_fps, device=self._device, dtype=torch.float32)
        self._motion_bodies = torch.stack(_motion_bodies).to(self._device).type(torch.float32)
        self._motion_aa = torch.tensor(np.concatenate(_motion_aa), device=self._device, dtype=torch.float32)

        self._motion_dt = torch.tensor(_motion_dt, device=self._device, dtype=torch.float32)
        self._motion_num_frames = torch.tensor(_motion_num_frames, device=self._device)
        self._num_motions = len(motions)
        
        self.gts = torch.cat([m.global_translation for m in motions], dim=0).float().to(self._device)
        self.grs = torch.cat([m.global_rotation for m in motions], dim=0).float().to(self._device)
        self.gavs = torch.cat([m.global_angular_velocity for m in motions], dim=0).float().to(self._device)
        self.gvs = torch.cat([m.global_velocity for m in motions], dim=0).float().to(self._device)
        self.dvs = torch.cat([m.dof_vels for m in motions], dim=0).float().to(self._device)
        
        if "global_translation_extend" in motions[0].__dict__:
            self.gts_t = torch.cat([m.global_translation_extend for m in motions], dim=0).float().to(self._device)
        
        if "dof_pos" in motions[0].__dict__:
            self.dof_pos = torch.cat([m.dof_pos for m in motions], dim=0).float().to(self._device)
        
        lengths = self._motion_num_frames
        lengths_shifted = lengths.roll(1)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)
        self.motion_ids = torch.arange(len(motions), dtype=torch.long, device=self._device)
        self.num_bodies = self.num_joints
        
        num_motions = self.num_motions()
        total_len = self.get_total_length()
        logger.info(f"Loaded {num_motions:d} motions with a total length of {total_len:.3f}s and {self.gts.shape[0]} frames.")
        return motions

    def load_motion_with_skeleton(self,
                                  motion_data_list,
                                  target_heading,
                                  ):
        # loading motion with the specified skeleton. Perfoming forward kinematics to get the joint positions
        res = {}
        for f in track(range(len(motion_data_list)), description="Loading motions..."):
            curr_file = motion_data_list[f]
            if not isinstance(curr_file, dict) and osp.isfile(curr_file):
                key = motion_data_list[f].split("/")[-1].split(".")[0]
                curr_file = joblib.load(curr_file)[key]

            seq_len = curr_file['root_trans_offset'].shape[0]
            start, end = 0, seq_len

            trans = to_torch(curr_file['root_trans_offset']).clone()[start:end]
            pose_aa = to_torch(curr_file['pose_aa'][start:end]).clone()
            
            dt = 1/curr_file['fps']

            if not target_heading is None and not self.align_method is None:

                if self.align_method == 'xyz':
                    logger.info('Aligning the root rotation on all directions')
                    start_root_rot = sRot.from_rotvec(pose_aa[0, 0])
                    heading_inv_rot = sRot.from_quat(calc_heading_quat_inv(torch.from_numpy(start_root_rot.as_quat()[None, ]), w_last=True))
                    heading_delta = sRot.from_quat(target_heading) * heading_inv_rot 
                    pose_aa[:, 0] = torch.tensor((heading_delta * sRot.from_rotvec(pose_aa[:, 0])).as_rotvec())

                    trans = torch.matmul(trans, torch.from_numpy(heading_delta.as_matrix().squeeze().T).float())
                
                elif self.align_method == 'xy':
                    logger.info('Aligning the root rotation on xy-plane')
                    cur_dir = sRot.from_rotvec(pose_aa[0, 0]).apply([1,0,0])
                    cur_dir[2] = 0
                    cur_dir/=np.linalg.norm(cur_dir)

                    tgt_dir = sRot.from_quat(target_heading).apply([1,0,0])
                    tgt_dir[2]=0
                    tgt_dir/=np.linalg.norm(tgt_dir)

                    angle = np.arccos(np.clip(np.dot(cur_dir, tgt_dir), -1.0, 1.0))
                    cross_prod = np.cross(cur_dir, tgt_dir)
                    if cross_prod[2]<0:
                        angle*=-1

                    rot_offset = sRot.from_rotvec([0,0,angle])
                    rots = sRot.from_rotvec(pose_aa[:,0])
                    pose_aa[:,0]=torch.from_numpy((rot_offset*rots).as_rotvec()).float()

                    trans = torch.matmul(trans.float(),torch.from_numpy(rot_offset.as_matrix().T).float())

                else:
                    logger.warning('You have assigned a wrong root rotation alignment method; Consider using xyz or xy; Abandon alignment!')
            
            if self.smooth_init:
                logger.info(f'Apply smoothing between the default state and the initial state! Interpolate {self.smooth_init_frames} in total!')
                
                extend_trans = trans[:1].repeat(self.smooth_init_frames, 1) # (smooth_frames, 3)
                trans = torch.cat([extend_trans, trans], dim=0)

                init_rot = pose_aa[:1]
                num_joint = init_rot.shape[1]
                init_quat = sRot.from_rotvec(init_rot.reshape(-1,3)).as_quat().reshape(1, num_joint, 4).repeat(self.smooth_init_frames, axis=0)

                default_rot = np.concatenate([np.array(self.default_angles)[:,None]*self.mesh_parsers.dof_axis.numpy(), np.zeros((len(self.m_cfg.extend_config), 3))], axis=0)
                assert num_joint-1 == default_rot.shape[0]
                default_quat = np.concatenate([target_heading[None,], sRot.from_rotvec(default_rot).as_quat()],axis=0)
                default_quat = default_quat[None,].repeat(self.smooth_init_frames, axis=0) # (smooth_frames, num_joint, 4)
                
                time_arr = torch.linspace(0, 1, self.smooth_init_frames).unsqueeze(-1).unsqueeze(-1)
            
                inter_quat = slerp(torch.from_numpy(default_quat), torch.from_numpy(init_quat), time_arr)

                inter_pose_aa = sRot.from_quat(inter_quat.reshape(-1,4)).as_rotvec().reshape(self.smooth_init_frames, num_joint, 3) # (smooth_frames, num_joints, 3)
                pose_aa = torch.cat([torch.from_numpy(inter_pose_aa), pose_aa], dim=0).float()

            if self.smooth_end:
                logger.info(f'Apply smoothing between the end state and the default state! Interpolate {self.smooth_end_frames} in total!')

                extend_trans = trans[-1:].repeat(self.smooth_end_frames, 1) # (smooth_frames, 3)
                trans = torch.cat([trans, extend_trans], dim=0)

                end_rot = pose_aa[-1:]
                num_joint = end_rot.shape[1]
                end_quat = sRot.from_rotvec(end_rot.reshape(-1, 3)).as_quat().reshape(1, num_joint, 4).repeat(self.smooth_end_frames, axis=0)
                
                default_rot = np.concatenate([np.dot(pose_aa[-1:, 0], np.array([0,0,1]))*np.array([0,0,1])[None,],np.array(self.default_angles)[:,None]*self.mesh_parsers.dof_axis.numpy(), np.zeros((len(self.m_cfg.extend_config), 3))], axis=0)
                assert num_joint == default_rot.shape[0]
                default_quat = sRot.from_rotvec(default_rot).as_quat()
                default_quat = default_quat[None,].repeat(self.smooth_end_frames, axis=0) # (smooth_frames, num_joint, 4)

                time_arr = torch.linspace(0, 1, self.smooth_end_frames).unsqueeze(-1).unsqueeze(-1)

                inter_quat = slerp(torch.from_numpy(end_quat), torch.from_numpy(default_quat), time_arr)

                inter_pose_aa = sRot.from_quat(inter_quat.reshape(-1,4)).as_rotvec().reshape(self.smooth_end_frames, num_joint, 3) # (smooth_frames, num_joints, 3)
                pose_aa = torch.cat([pose_aa, torch.from_numpy(inter_pose_aa)], dim=0).float()


            if self.mesh_parsers is not None:
                curr_motion = self.mesh_parsers.fk_batch(pose_aa[None, ], trans[None, ], return_full= True, dt = dt)
                curr_motion = EasyDict({k: v.squeeze() if torch.is_tensor(v) else v for k, v in curr_motion.items() })
                res[f] = (curr_file, curr_motion)
            else:
                logger.error("No mesh parser found")
        return res
    

    def num_motions(self):
        return self._num_motions


    def get_total_length(self):
        return sum(self._motion_lengths)

    def get_motion_length(self, motion_ids=None):
        if motion_ids is None:
            return self._motion_lengths
        else:
            return self._motion_lengths[motion_ids]


    def _calc_frame_blend(self, time, len, num_frames, dt):
        time = time.clone()
        phase = time / len
        phase = torch.clip(phase, 0.0, 1.0)  # clip time to be within motion length.
        time[time < 0] = 0

        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = torch.clip((time - frame_idx0 * dt) / dt, 0.0, 1.0) # clip blend to be within 0 and 1
        
        return frame_idx0, frame_idx1, blend