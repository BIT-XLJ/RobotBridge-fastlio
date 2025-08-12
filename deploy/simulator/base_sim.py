import numpy as np
from loguru import logger
class BaseSim:
    def __init__(self, config):
        self.cfg = config
        self._setup()
        self._load_asset()
        self._init_low_state()

    def _setup(self):
        self.low_dt = self.cfg.control.low_dt
        self.decimation = self.cfg.control.decimation
        self.high_dt = self.low_dt * self.decimation

        logger.info(f'Robot-level Control Frequency Set to {1/self.low_dt}HZ')
        logger.info(f'Policy-level Control Frequency Set to {1/self.high_dt}HZ')

    def _load_asset(self):
        self.default_angles = np.array(self.cfg.asset.default_angles, dtype=np.float32)
        self.kps = np.array(self.cfg.asset.kps, dtype=np.float32)
        self.kds = np.array(self.cfg.asset.kds, dtype=np.float32)
        self.dof_names = list(self.cfg.asset.joint_order.keys())
        self.frozen_dof_names = self.cfg.asset.frozen_dof_names
        self.active_dof_idx = np.array([i for i in range(len(self.dof_names)) if self.dof_names[i] not in self.frozen_dof_names])
        self.num_dof = self.cfg.asset.num_dof
        self.num_action = self.cfg.asset.num_action
        
        for i,k in enumerate(self.dof_names):
            logger.info(f'Joint {k}  Default Angles {self.default_angles[i]}  P_gain {self.kps[i]}  D_gain {self.kds[i]}  Frozen {k in self.frozen_dof_names}')
        logger.info(f'Total Number of dof: {self.num_dof}')
        logger.info(f'Number of Action: {self.num_action}')

    def _init_low_state(self):
        self.root_trans = np.zeros((1,3), dtype=np.float32)
        self.root_quat = np.zeros((1,4), dtype=np.float32)
        self.root_rpy = np.zeros((1,3), dtype=np.float32)
        self.base_lin_vel = np.zeros((1,3), dtype=np.float32)
        self.base_ang_vel = np.zeros((1,3),dtype=np.float32)
        self.projected_gravity = np.zeros((1,3), dtype=np.float32)
        self.dof_pos = np.zeros((1, self.num_action), dtype=np.float32)
        self.dof_vel = np.zeros((1, self.num_action), dtype=np.float32)
        self.marker_pos = None

    def update_obs(self):
        self._refresh_sim()

        return {
            'root_trans': self.root_trans,
            'base_lin_vel': self.base_lin_vel, # in robot local coordinate
            'root_quat': self.root_quat,
            'root_rpy': self.root_rpy,
            'base_ang_vel': self.base_ang_vel,
            'dof_pos': self.dof_pos,
            'dof_vel': self.dof_vel,
            'projected_gravity': self.projected_gravity,
        }
    
    def update_marker_pos(self, marker_pos):
        self.marker_pos = marker_pos.squeeze(0)
    
    def check_termination(self):
        raise NotImplementedError

    def calibrate(self, refresh):
        raise NotImplementedError
    
    def apply_action(self, action):
        raise NotImplementedError
    
    def _refresh_sim(self):
        raise NotImplementedError