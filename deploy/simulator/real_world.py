import time
import lcm
import copy
import select
import threading
import numpy as np
import pickle
from simulator.base_sim import BaseSim
from loguru import logger

import sys
sys.path.append('../')
from unitree_sdk2.lcm_types.body_control_data_lcmt import body_control_data_lcmt
from unitree_sdk2.lcm_types.rc_command_lcmt import rc_command_lcmt
from unitree_sdk2.lcm_types.state_estimator_lcmt import state_estimator_lcmt
from unitree_sdk2.lcm_types.pd_tau_targets_lcmt import pd_tau_targets_lcmt

from utils.helpers import get_gravity, get_rpy
from utils.pos_server import fast_lio_odometry_lcmt
from utils.local2world import fk_dof
from utils.torch_utils import my_quat_rotate
import numpy as np

# 定义fast_lio里程计LCM消息类型
# class fast_lio_odometry_lcmt:
#     def __init__(self):
#         self.p = [0.0, 0.0, 0.0]  # position x, y, z
#         self.quat = [0.0, 0.0, 0.0, 1.0]  # quaternion x, y, z, w
    
#     def encode(self):
#         data = {
#             'p': self.p,
#             'quat': self.quat
#         }
#         return pickle.dumps(data)
    
#     @classmethod
#     def decode(cls, data):
#         decoded_data = pickle.loads(data)
#         msg = cls()
#         msg.p = decoded_data['p']
#         msg.quat = decoded_data['quat']
#         return msg

class RealWorld(BaseSim):
    def __init__(self, config):
        super().__init__(config)
        self._init_communication()
        self.spin()
        while True:
            if self.connected:
                break
    
    def _setup(self):
        super()._setup()
        self.lc = lcm.LCM('udpm://239.255.76.67:7667?ttl=255')
    
    def _load_asset(self):
        super()._load_asset()
        self.joint_serial_num = self.cfg.asset.joint_order
        self.policy_joint_order = list(self.joint_serial_num.keys())
        # index for policy to robot
        self.idx_r2p=[self.joint_serial_num[k] for k in self.joint_serial_num]
        # index for robot to policy
        self.idx_p2r=[self.idx_r2p.index(i) for i in range(self.num_dof)]

    def _init_low_state(self):
        super()._init_low_state()
        
        # remote controller part
        self.mode = 0
        self.ctrlmode_left = 0
        self.ctrlmode_right = 0
        self.left_stick = [0, 0]
        self.right_stick = [0, 0]
        self.left_upper_switch = 0
        self.left_lower_left_switch = 0
        self.left_lower_right_switch = 0
        self.right_upper_switch = 0
        self.right_lower_left_switch = 0
        self.right_lower_right_switch = 0
        self.left_upper_switch_pressed = 0
        self.left_lower_left_switch_pressed = 0
        self.left_lower_right_switch_pressed = 0
        self.right_upper_switch_pressed = 0
        self.right_lower_left_switch_pressed = 0
        self.right_lower_right_switch_pressed = 0
    
    def _init_communication(self):
        self.firstReceiveAlarm = False
        self.firstReceiveOdometer = False
        self.firstReceiveFastLIO = False
        self._init_time = time.time()
        
        self.root_state_subscriber = self.lc.subscribe('state_estimator_data', self._root_state_handler)
        self.joint_state_subscriber = self.lc.subscribe('body_control_data', self._joint_state_handler)
        self.remote_controller_subscriber = self.lc.subscribe('rc_command_data', self._remote_controller_handler)
        # 添加对fast_lio里程计的订阅
        self.fast_lio_subscriber = self.lc.subscribe('fast_lio_odometry', self._fast_lio_handler)
        
        # 初始化fast_lio数据
        self.fast_lio_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.fast_lio_quat = np.array([0., 0., 0., 1.], dtype=np.float32)
        self.Odom_offset = np.array([0.0, 0.0, 0.0], dtype=np.float32)        #这里要记得修改

    def _root_state_handler(self, channel, data):
        msg = state_estimator_lcmt.decode(data)
        if not self.firstReceiveOdometer:
            self.firstReceiveOdometer = True
            logger.info('State Estimator Information Received!')
            logger.info(f'Root Translation: {np.array(msg.p)}, Root Linear Velocity: {np.array(msg.vBody)}')
        self.root_trans_tmp = np.array(msg.p)
        self.root_rpy_tmp = np.array(msg.rpy)
        self.root_quat_tmp = np.roll(np.array(msg.quat), -1) # (w,x,y,z) -> (x,y,z,w)
        self.base_lin_vel_tmp = np.array(msg.vBody)
        self.base_ang_vel_tmp = np.array(msg.omegaBody)

    def _joint_state_handler(self, channel, data):
        msg = body_control_data_lcmt.decode(data)
        if not self.firstReceiveAlarm:
            self.time_delay = time.time() - self._init_time
            self.firstReceiveAlarm = True
            logger.info("Communication build successfully between the policy and the transition layer!")
            logger.info(f'First signal arrives after {self.time_delay}s!')
        self.dof_pos_tmp = np.array(msg.q)[self.idx_r2p]
        self.dof_vel_tmp = np.array(msg.qd)[self.idx_r2p]
    
    def _remote_controller_handler(self, channel, data):
        msg = rc_command_lcmt.decode(data)
        
        self.left_upper_switch_pressed = ((msg.left_upper_switch and not self.left_upper_switch) or self.left_upper_switch_pressed)
        self.left_lower_left_switch_pressed = ((msg.left_lower_left_switch and not self.left_lower_left_switch) or self.left_lower_left_switch_pressed)
        self.left_lower_right_switch_pressed = ((msg.left_lower_right_switch and not self.left_lower_right_switch) or self.left_lower_right_switch_pressed)
        self.right_upper_switch_pressed = ((msg.right_upper_switch and not self.right_upper_switch) or self.right_upper_switch_pressed)
        self.right_lower_left_switch_pressed = ((msg.right_lower_left_switch and not self.right_lower_left_switch) or self.right_lower_left_switch_pressed)
        self.right_lower_right_switch_pressed = ((msg.right_lower_right_switch) and not self.right_lower_right_switch) or self.right_lower_right_switch_pressed

        self.mode = msg.mode
        self.right_stick = msg.right_stick
        self.left_stick = msg.left_stick
        self.left_upper_switch = msg.left_upper_switch
        self.left_lower_left_switch = msg.left_lower_left_switch
        self.left_lower_right_switch = msg.left_lower_right_switch
        self.right_upper_switch = msg.right_upper_switch
        self.right_lower_left_switch = msg.right_lower_left_switch
        self.right_lower_right_switch = msg.right_lower_right_switch

    def _fast_lio_handler(self, channel, data):
        try:
            msg = fast_lio_odometry_lcmt.decode(data)
            if not self.firstReceiveFastLIO:
                self.firstReceiveFastLIO = True
                logger.info('FastLIO Odometry Information Received!')
                logger.info(f'Position: {np.array(msg.position)}, Quaternion: {np.array(msg.quaternion)}')
            
            self.fast_lio_position = np.array(msg.position, dtype=np.float32)
            self.fast_lio_quat = np.array(msg.quaternion, dtype=np.float32)
        except Exception as e:
            logger.error(f"Error decoding FastLIO message: {e}")
            import traceback
            traceback.print_exc()

    def _refresh_sim(self):
        self.root_trans = self.root_trans_tmp.copy()
        self.base_lin_vel = self.base_lin_vel_tmp.copy()
        self.root_quat = self.root_quat_tmp.copy()
        self.root_rpy = self.root_rpy_tmp.copy()
        self.base_ang_vel = self.base_ang_vel_tmp.copy()
        self.projected_gravity = get_gravity(self.root_quat, w_last=True)
        self.dof_pos = self.dof_pos_tmp.copy()[self.active_dof_idx]        #把这里的dof_pos直接丢给fk_dof就可以
        self.dof_vel = self.dof_vel_tmp.copy()[self.active_dof_idx]
        
        # 更新fast_lio数据（如果可用）
        if hasattr(self, 'fast_lio_position') and hasattr(self, 'fast_lio_quat'):
            # 这里可以根据需要选择使用哪种定位数据
            # 例如：self.root_trans = self.fast_lio_position.copy()
            # 在这里使用fast_lio的雷达里程计信息，结合FK，更新root_trans，和base_lin_vel

            pass

    def apply_action(self, action):
        action = action.squeeze(0)
        dof_target_pos = self.default_angles.copy()
        dof_target_pos[self.active_dof_idx]+=action*self.cfg.control.action_scale
        dof_target_pos = dof_target_pos[self.idx_p2r]

        cmd = pd_tau_targets_lcmt()
        cmd.q_des = dof_target_pos.copy()
        cmd.qd_des = np.zeros_like(dof_target_pos)
        cmd.kp = self.kps.copy()[self.idx_p2r]
        cmd.kd = self.kds.copy()[self.idx_p2r]
        cmd.tau_ff = np.zeros_like(dof_target_pos)
        cmd.se_contactState = np.zeros(2)
        cmd.timestamp_us = int(time.time()*10**6)
        
        self.lc.publish("pd_plustau_targets", cmd.encode())

    def poll(self, cb=None):
        t = time.time()
        try:
            while True:
                timeout = 0.01
                rfds, wfds, efds = select.select([self.lc.fileno()], [], [], timeout)
                if rfds:
                    self.lc.handle()
                else:
                    continue
        except KeyboardInterrupt:
            pass

    def spin(self):
        self.run_thread = threading.Thread(target=self.poll, daemon=False)
        self.run_thread.start()

    def close(self):
        self.lc.unsubscribe(self.joint_state_subscriber)
        self.lc.unsubscribe(self.fast_lio_subscriber)

    def connected(self):
        return self.firstReceiveAlarm
    
    def calibrate(self, refresh):
        self._refresh_sim()
        if refresh:
            logger.info('Calibraiting..., Press R2 to continue')
            while True:
                if self.right_lower_right_switch_pressed:
                    logger.info('R2 button pressed, Start Calibrating...')
                    self.right_lower_right_switch_pressed = False
                    break

            cur_dof_pos = self.dof_pos
            final_goal = np.zeros_like(self.default_angles)[self.active_dof_idx]
            default_pos = self.default_angles.copy()[self.active_dof_idx]
            target = cur_dof_pos - default_pos
            target_seq = []
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
            logger.info('Calibration Done. Press R2 to continue')
            while True:
                if self.right_lower_right_switch_pressed:
                    logger.info('R2 pressed again, Communication built between policy layer and transition layer!')
                    self.right_lower_right_switch_pressed =  False
                    break
                
        else:
            raise NotImplementedError
            


    def check_termination(self):
        return abs(self.root_rpy_tmp[0])>0.8 or abs(self.root_rpy_tmp[1])>0.8 or self.right_lower_right_switch_pressed

    def compute_fk_body_pos(joint_pos: np.ndarray):
        extend_body_parent_ids = [22, 29, 15]
        extend_body_pos = np.array([[0.06, 0, 0], [0.06, 0, 0], [0, 0, 0.4]])

        body_pos, body_quat = fk_dof(np.from_numpy(joint_pos))
        body_quat = body_quat.roll(shifts=-1, dims=-1)

        extend_curr_pos = my_quat_rotate(body_quat[extend_body_parent_ids].reshape(-1, 4),
                                                            extend_body_pos.reshape(-1, 3)).view(
            -1, 3) + body_pos[extend_body_parent_ids]
        body_pos_extend = np.cat([body_pos, extend_curr_pos], dim=0)
        body_quat_extend = np.cat([body_quat, body_quat[extend_body_parent_ids[0]:extend_body_parent_ids[0]+1, :],
                                  body_quat[extend_body_parent_ids[1]:extend_body_parent_ids[1]+1, :],
                                  body_quat[extend_body_parent_ids[2]:extend_body_parent_ids[2]+1, :]])
        return body_pos_extend, body_quat_extend
        
            