import lcm
from collections import deque
import pickle
import numpy as np
import time
import select

from multiprocessing import Process, Queue

import rospy
from nav_msgs.msg import Odometry

# 定义LCM消息类型，用于发送位置和姿态信息
class fast_lio_odometry_lcmt:
    def __init__(self):
        self.position = [0.0, 0.0, 0.0]  # x, y, z
        self.quaternion = [0.0, 0.0, 0.0, 1.0]  # x, y, z, w
    
    def encode(self):
        # 简单的编码方式，将数据打包成字节
        data = {
            'position': self.position,
            'quaternion': self.quaternion
        }
        return pickle.dumps(data)
    
    @classmethod
    def decode(cls, data):
        # 解码方式
        decoded_data = pickle.loads(data)
        msg = cls()
        msg.position = decoded_data['position']
        msg.quaternion = decoded_data['quaternion']
        return msg


class Position_Node:
    def __init__(self, lcm_instance):
        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.quat = np.array([0., 0., 0., 1.], dtype=np.float32)    # xyzw
        self.lcm = lcm_instance

    def callback(self, data):
        rospy.loginfo(data.pose.pose.position)
        rospy.loginfo(data.pose.pose.orientation)
        self.position = np.array([data.pose.pose.position.x, data.pose.pose.position.y, data.pose.pose.position.z], dtype=np.float32)
        self.quat = np.array([data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w], dtype=np.float32)
        
        # 创建LCM消息并发送
        try:
            msg = fast_lio_odometry_lcmt()
            msg.position = self.position.tolist()
            msg.quaternion = self.quat.tolist()
            
            # 发送到LCM频道
            self.lcm.publish("fast_lio_odometry", msg.encode())
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error sending LCM message: {e}")

    def localization_listener(self):
        rospy.init_node('localization_listener', anonymous=True)
        rospy.Subscriber('/Odometry', Odometry, self.callback)     #fast_lio2的topic是/Odometry
        rospy.spin()

    def main_loop(self):
        self.localization_listener()

    @staticmethod
    def start_main_loop(lcm_instance):
        node = Position_Node(lcm_instance)
        node.main_loop()


class Position_Server:
    def __init__(self, config: dict = None, Unit_Test = False):

        # 初始化LCM
        self.lcm = lcm.LCM('udpm://239.255.76.67:7667?ttl=255')
        
        print("[Position Server] Position server has started, waiting for client connections...")

    def _close(self):
        self.lcm.close()
        print("[Position Server] The server has been closed.")

    def send_process(self):
        # 启动ROS节点进程
        process = Process(target=Position_Node.start_main_loop, args=(self.lcm,))
        process.start()
        
        try:
            # 主循环处理LCM消息
            while True:
                # 处理LCM消息
                timeout = 0.01
                rfds, wfds, efds = select.select([self.lcm.fileno()], [], [], timeout)
                if rfds:
                    self.lcm.handle()
                time.sleep(0.001)  # 小延迟避免CPU占用过高
        except KeyboardInterrupt:
            print("[Position Server] Interrupted by user.")
        finally:
            self._close()
        process.terminate()
        process.join()


if __name__ == "__main__":
    pos_server = Position_Server()
    pos_server.send_process()
