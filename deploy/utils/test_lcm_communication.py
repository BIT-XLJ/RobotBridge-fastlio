#!/usr/bin/env python3
"""
测试LCM通信的脚本
用于验证pos_server.py和real_world.py之间的通信
"""

import lcm
import time
import pickle
import numpy as np

# 定义消息类型（与pos_server.py中保持一致）
class position_odometry_lcmt:
    def __init__(self):
        self.position = [0.0, 0.0, 0.0]  # x, y, z
        self.quaternion = [0.0, 0.0, 0.0, 1.0]  # x, y, z, w
    
    def encode(self):
        data = {
            'position': self.position,
            'quaternion': self.quaternion
        }
        return pickle.dumps(data)

def test_lcm_publisher():
    """测试LCM发布者"""
    lc = lcm.LCM('udpm://239.255.76.67:7667?ttl=255')
    
    print("开始发送测试数据...")
    
    try:
        for i in range(10):
            # 创建测试消息
            msg = position_odometry_lcmt()
            msg.position = [float(i), float(i), 0.0]  # 模拟移动
            msg.quaternion = [0.0, 0.0, np.sin(i*0.1), np.cos(i*0.1)]  # 模拟旋转
            
            # 发送消息
            lc.publish("fast_lio_odometry", msg.encode())
            print(f"发送消息 {i+1}: 位置={msg.position}, 四元数={msg.quaternion}")
            
            time.sleep(1.0)  # 每秒发送一次
            
    except KeyboardInterrupt:
        print("\n停止发送")
    finally:
        lc.close()

def test_lcm_subscriber():
    """测试LCM订阅者"""
    lc = lcm.LCM('udpm://239.255.76.67:7667?ttl=255')
    
    def message_handler(channel, data):
        try:
            # 解码消息
            decoded_data = pickle.loads(data)
            print(f"收到消息: {decoded_data}")
        except Exception as e:
            print(f"解码消息失败: {e}")
    
    # 订阅频道
    subscription = lc.subscribe("fast_lio_odometry", message_handler)
    
    print("开始监听LCM消息...")
    print("按Ctrl+C停止")
    
    try:
        while True:
            lc.handle()
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n停止监听")
    finally:
        lc.unsubscribe(subscription)
        lc.close()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "sub":
        test_lcm_subscriber()
    else:
        test_lcm_publisher() 