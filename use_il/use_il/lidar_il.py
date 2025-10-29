#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import numpy as np
import torch
import torch.nn as nn

# 跟訓練一致的 MLP
class MLP(nn.Module):
    def __init__(self, bins=360):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(bins, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
    def forward(self, x):
        return self.net(x)

class ILLidarNode(Node):
    def __init__(self):
        super().__init__('il_lidar_node')
        # 參數（可用 ros2 param 改）
        self.declare_parameter('model_path', '/home/open/ws_navlearn/models/il_lidar.pt')
        self.declare_parameter('bins', 360)
        self.declare_parameter('max_range', 3.5)
        self.declare_parameter('topic_scan', '/scan')
        self.declare_parameter('topic_cmd',  '/cmd_vel')
        self.declare_parameter('v_max', 10.35)      # 封頂速度，先保守
        self.declare_parameter('w_max', 10.5)
        self.declare_parameter('safety_stop', 0.28)# 最近距離 < 0.28m 就急停
        self.declare_parameter('hz', 15.0)         # 最多每秒發 15 次（節流）

        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.bins       = int(self.get_parameter('bins').value)
        self.max_range  = float(self.get_parameter('max_range').value)
        self.v_max      = float(self.get_parameter('v_max').value)
        self.w_max      = float(self.get_parameter('w_max').value)
        self.safety_stop= float(self.get_parameter('safety_stop').value)
        self.hz         = float(self.get_parameter('hz').value)

        # 載模型
        ckpt = torch.load(self.model_path, map_location='cpu')
        self.model = MLP(bins=ckpt.get('bins', self.bins))
        self.model.load_state_dict(ckpt['model'])
        # 我們現在是考試模式，不是訓練模式
        self.model.eval()
        self.model_bins  = ckpt.get('bins', self.bins)
        self.model_range = ckpt.get('max_range', self.max_range)
        self.get_logger().info(f'Loaded model: {self.model_path} (bins={self.model_bins}, max_range={self.model_range})')

        # 訂閱 /scan、發布 /cmd_vel
        self.pub = self.create_publisher(Twist, self.get_parameter('topic_cmd').value, 10)
        self.sub = self.create_subscription(LaserScan, self.get_parameter('topic_scan').value, self.cb_scan, 10)

        # 節流：控制每秒最多輸出 hz 次
        self.last_pub_time = self.get_clock().now()

    def preprocess(self, ranges):
        arr = np.array(ranges, dtype=np.float32)
        # 把無限/NaN 變成 max_range
        arr[~np.isfinite(arr)] = self.max_range
        # clip 到 [0, max_range]，再正規化到 [0,1]
        arr = np.clip(arr, 0.0, self.max_range) / self.max_range
        # 重採樣到訓練 bins
        if arr.shape[0] != self.model_bins:
            x_src = np.linspace(0.0, 1.0, arr.shape[0], dtype=np.float32)
            x_dst = np.linspace(0.0, 1.0, self.model_bins, dtype=np.float32)
            arr = np.interp(x_dst, x_src, arr).astype(np.float32)
        return arr

    def cb_scan(self, msg: LaserScan):
        # 節流
        now = self.get_clock().now()
        if (now - self.last_pub_time).nanoseconds < 1e9 / max(1.0, self.hz):
            return

        # 最近距離（安全急停判斷用）
        raw = np.array(msg.ranges, dtype=np.float32)
        raw[~np.isfinite(raw)] = self.max_range
        d_min = float(np.min(raw))

        # 前處理
        x = self.preprocess(msg.ranges)  # [model_bins] in [0,1]
        x_t = torch.from_numpy(x).unsqueeze(0)     # [1, bins]

        # 推論
        with torch.no_grad():
            y = self.model(x_t).squeeze(0).numpy()  # [2] → (v, w)
        v, w = float(y[0]), float(y[1])

        # 速度封頂（避免太快）
        v = float(np.clip(v, -self.v_max, self.v_max))
        w = float(np.clip(w, -self.w_max, self.w_max))

        # 安全層：太近就停（或只允許轉向）
        if d_min < self.safety_stop:
            v = 0.0
            # 可選：若正前方很近，根據左右差讓 w 稍微避一下
            # 這裡先簡單處理，只停車
            self.get_logger().warn(f'safety stop: d_min={d_min:.2f} < {self.safety_stop:.2f}')

        # 發指令
        tw = Twist()
        tw.linear.x  = v
        tw.angular.z = w
        self.pub.publish(tw)
        self.last_pub_time = now

    def destroy_node(self):
        # 停車
        tw = Twist()
        self.pub.publish(tw)
        super().destroy_node()

def main():
    rclpy.init()
    node = ILLidarNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

