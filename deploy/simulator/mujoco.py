import os
import time
import copy
import numpy as np
import torch
from simulator.base_sim import BaseSim
import mujoco.viewer
import mujoco
from utils.helpers import get_gravity, get_rpy
from utils.motion_lib.rotations import calc_heading_quat_inv, my_quat_rotate
from loguru import logger
from scipy.spatial.transform import Rotation as sRot

class Mujoco(BaseSim):
    def __init__(self, config):
        super().__init__(config)
        self.marker = self.cfg.get('marker', False)
        self.random_init_heading = self.cfg.get('random_init_heading', False)
        logger.info(f'Visualization Marker: {self.marker}')
        self._load_viewer()

    def _load_asset(self):
        super()._load_asset()
        xml_path = os.path.join(self.cfg.asset.asset_root, self.cfg.asset.asset_file)
        
        self.mujoco_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mujoco_data = mujoco.MjData(self.mujoco_model)
        self.mujoco_model.opt.timestep=self.low_dt
        
        self.default_qpos = self.mujoco_data.qpos.copy()
        self.default_qvel = self.mujoco_data.qvel.copy()

    def _load_viewer(self):
        self.viewer = mujoco.viewer.launch_passive(self.mujoco_model, self.mujoco_data)
        self.marker_pos = None

    def render(self):
        if self.marker_pos is not None:
            self.viewer.user_scn.ngeom = 0
            for i in range(self.marker_pos.shape[0]):
                mujoco.mjv_initGeom(
                    self.viewer.user_scn.geoms[i],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.03, 0, 0],
                    pos=self.marker_pos[i],
                    mat=np.eye(3).flatten(),
                    rgba=np.array([1,0,0,1])
                )
            self.viewer.user_scn.ngeom=self.marker_pos.shape[0]
        self.viewer.sync()

    def _refresh_sim(self):
        self.root_trans = self.mujoco_data.qpos[:3]
        self.root_quat = np.roll(self.mujoco_data.qpos[3:7], -1) # (w,x,y,z) -> (x,y,z,w)
        heading_rot_inv = calc_heading_quat_inv(torch.from_numpy(self.root_quat).unsqueeze(0), w_last=True)
        self.base_lin_vel = my_quat_rotate(heading_rot_inv,  torch.from_numpy(self.mujoco_data.qvel[:3]).unsqueeze(0)).numpy().squeeze(0)
        self.root_rpy = get_rpy(self.root_quat, w_last=True)
        self.projected_gravity = get_gravity(self.root_quat, w_last=True)
        self.base_ang_vel = self.mujoco_data.qvel[3:6] # local
        self.dof_pos = self.mujoco_data.qpos[7:]
        self.dof_vel = self.mujoco_data.qvel[6:]
    
    def apply_action(self, action):
        action = action.squeeze(0)
        torque_limit = np.array(self.cfg.control.torque_clip_value, dtype=np.float32)
        self.render()
        for _ in range(self.decimation):
            tgt_dof_pos = self.default_angles[self.active_dof_idx].copy() + action*self.cfg.control.action_scale
            torque = (tgt_dof_pos - self.mujoco_data.qpos[7:])*self.kps[self.active_dof_idx] - self.mujoco_data.qvel[6:]*self.kds[self.active_dof_idx]
            torque = np.clip(torque, -torque_limit, torque_limit)
            self.mujoco_data.ctrl[:] = torque
            mujoco.mj_step(self.mujoco_model, self.mujoco_data)

    def calibrate(self, refresh):
        if refresh:
            default_qpos = self.default_qpos.copy()
            default_qvel = self.default_qvel
            default_qpos[7:] = self.default_angles[self.active_dof_idx]

            if self.random_init_heading:
                logger.info('Randomize Initial Heading for Testing Robustness')
                cur_dir = sRot.from_quat(np.roll(default_qpos[3:7],-1)).apply([1,0,0])
                cur_dir[2]=0
                cur_dir/=np.linalg.norm(cur_dir)

                tgt_dir = 2*np.random.random((3,))-1
                tgt_dir[2]=0
                tgt_dir/=np.linalg.norm(tgt_dir)
                
                angle=np.arccos(np.clip(np.dot(cur_dir, tgt_dir), -1.0, 1.0))
                cross_prod = np.cross(cur_dir, tgt_dir)
                if cross_prod[2]<0:
                    angle*=-1

                rot_offset=sRot.from_rotvec([0,0, angle])
                rots=sRot.from_quat(np.roll(default_qpos[3:7], -1))
                default_qpos[3:7]= np.roll((rot_offset*rots).as_quat(),1)
                logger.info(f'Current Facing Direction: {(rot_offset*rots).apply([1,0,0])}')
            
            self.mujoco_data.qpos[:] = default_qpos.copy()
            self.mujoco_data.qvel[:] = default_qvel
            self.mujoco_data.ctrl[:] = 0

        else:
            self._refresh_sim()
            cur_dof_pos = self.dof_pos
            final_goal = np.zeros_like(self.default_angles[self.active_dof_idx])
            default_pos = self.default_angles[self.active_dof_idx]
            target = cur_dof_pos - default_pos
            target_seq=[]
            while np.max(np.abs(target-final_goal))>0.01:
                target-=np.clip((target-final_goal), -0.05, 0.05)
                target_seq += [copy.deepcopy(target)]
            for tgt in target_seq:
                step_start=time.time()
                next_tgt = tgt/self.cfg.control.action_scale
                self.apply_action(next_tgt[None,])
                self._refresh_sim()
                time_till_next_step = self.high_dt - (time.time()-step_start)
                if time_till_next_step>0:
                    time.sleep(time_till_next_step)
            logger.info(f'Simulation Done!')
            while True:
                st = input('Press n to continue...')
                if 'n' in st:
                    break
                step_start=time.time()
                self.apply_action(np.zeros((1, self.num_action)))
                time_till_next_step = self.high_dt - (time.time()-step_start)
                if time_till_next_step>0:
                    time.sleep(time_till_next_step)

    
    def check_termination(self):
        return abs(self.root_rpy[0])>0.8 or abs(self.root_rpy[1])>0.8