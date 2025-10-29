# -*- coding: utf-8 -*-
"""
IL（模仿學習）LiDAR 避障節點：內建「人工接管優先」版
------------------------------------------------
差異點：
- 內建一個「小型 mux」：訂閱 /cmd_vel/manual，只要最近 manual_timeout 秒內收到手動，
  就採用手動；否則採用模型輸出（v̂, ŵ）。
- 仍保留：安全層（太近只允許轉向）、增益/最小速度地板、節流 log、CSV 記錄。

啟動建議：
- topic_cmd 設為 /cmd_vel（因為不再用 twist_mux）
- manual_topic 設為 /cmd_vel/manual
- manual_timeout 依你手感 0.2~0.5s
"""

import os, time, csv, math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

import torch
import torch.nn as nn

# ---------- MLP 結構要與訓練時一致 ----------
class MLP(nn.Module):
    def __init__(self, bins=360):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(bins, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # (v, w)
        )
    def forward(self, x):
        return self.net(x)

class LidarILNode(Node):
    def __init__(self):
        super().__init__('il_lidar_node')

        # ====== 可調參數（ROS 參數） ======
        # Topic 與模型
        self.declare_parameter('topic_scan', '/scan')            # LiDAR topic
        self.declare_parameter('topic_cmd',  '/cmd_vel')          # 最終輸出（不再用 mux，直接 /cmd_vel）
        self.declare_parameter('model_path', '~/ws_navlearn/models/il_lidar.pt')

        # 接管（手動遙控）
        self.declare_parameter('manual_topic',  '/cmd_vel/manual')  # 手動遙控來這裡
        self.declare_parameter('manual_timeout', 0.30)              # 最近 T 秒內有手動 → 用手動

        # 前處理
        self.declare_parameter('bins', 360)
        self.declare_parameter('max_range', 3.5)
        self.declare_parameter('hz', 15.0)

        # 安全/限幅
        self.declare_parameter('safety_stop', 0.28)
        # self.declare_parameter('v_max', 0.4)
        self.declare_parameter('v_max', 2.4)
        self.declare_parameter('w_max', 1.2)

        # 急停但允許轉向
        self.declare_parameter('slowturn_gain', 2.0)

        # 增益 + 最小速度地板（可用來「敢走一點」；不想用就都設 1.0 / 0.0）
        self.declare_parameter('v_gain', 1.0)
        self.declare_parameter('w_gain', 1.0)
        self.declare_parameter('v_min',  0.0)
        self.declare_parameter('w_min',  0.0)

        # 記錄
        self.declare_parameter('log_rate', 2.0)                   # 每秒最多印幾次
        self.declare_parameter('log_csv',  '')                    # 若非空字串就寫 CSV

        # 取值
        self.topic_scan   = self.get_parameter('topic_scan').get_parameter_value().string_value
        self.topic_cmd    = self.get_parameter('topic_cmd').get_parameter_value().string_value
        self.model_path   = os.path.expanduser(self.get_parameter('model_path').get_parameter_value().string_value)
        self.manual_topic = self.get_parameter('manual_topic').get_parameter_value().string_value
        self.manual_to    = float(self.get_parameter('manual_timeout').value)

        self.bins        = int(self.get_parameter('bins').value)
        self.max_range   = float(self.get_parameter('max_range').value)
        self.hz          = float(self.get_parameter('hz').value)

        self.safety_stop = float(self.get_parameter('safety_stop').value)
        self.v_max       = float(self.get_parameter('v_max').value)
        self.w_max       = float(self.get_parameter('w_max').value)
        self.slowturn_gain = float(self.get_parameter('slowturn_gain').value)

        self.v_gain      = float(self.get_parameter('v_gain').value)
        self.w_gain      = float(self.get_parameter('w_gain').value)
        self.v_min       = float(self.get_parameter('v_min').value)
        self.w_min       = float(self.get_parameter('w_min').value)

        self.log_rate    = float(self.get_parameter('log_rate').value)
        self.log_csv     = self.get_parameter('log_csv').get_parameter_value().string_value
        self._last_log_t = 0.0
        self._csv_path   = os.path.expanduser(self.log_csv) if self.log_csv else ''

        if self._csv_path:
            os.makedirs(os.path.dirname(self._csv_path), exist_ok=True)
            if not os.path.exists(self._csv_path):
                with open(self._csv_path, 'w', newline='') as f:
                    w = csv.writer(f)
                    w.writerow(['stamp_ns','d_min','mode','v_hat','w_hat','v_cmd','w_cmd'])

        # ====== 載入模型 ======
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.get_logger().info(f'loading model: {self.model_path} (map_location={device})')

        ckpt = torch.load(self.model_path, map_location=device)
        state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
        self.model = MLP(bins=self.bins).to(self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        self.get_logger().info('model ready.')

        # ====== ROS 通訊 ======
        self.sub_scan   = self.create_subscription(LaserScan, self.topic_scan, self.cb_scan, 10)
        self.sub_manual = self.create_subscription(Twist,     self.manual_topic, self.cb_manual, 10)
        self.pub_cmd    = self.create_publisher(Twist, self.topic_cmd, 10)

        self.last_scan = None
        self.last_manual = None           # 最近一次手動 Twist（或 None）
        self.last_manual_time = 0.0       # 最近手動的時間戳（time.time()）

        self.timer = self.create_timer(1.0 / max(self.hz, 1e-3), self.on_timer)

    # -------- 手動遙控回呼：只存最新，記下時間 --------
    def cb_manual(self, msg: Twist):
        self.last_manual = msg
        self.last_manual_time = time.time()

    # -------- LiDAR 回呼：做清理、重採樣、正規化 --------
    def cb_scan(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=np.float32)
        rng_max = np.float32(self.max_range if msg.range_max == 0.0 else min(msg.range_max, self.max_range))
        bad = ~np.isfinite(ranges)
        ranges[bad] = rng_max
        ranges = np.clip(ranges, 0.0, rng_max)

        if ranges.shape[0] != self.bins:
            x_src = np.linspace(0.0, 1.0, ranges.shape[0], dtype=np.float32)
            x_dst = np.linspace(0.0, 1.0, self.bins, dtype=np.float32)
            ranges = np.interp(x_dst, x_src, ranges).astype(np.float32)

        self.last_scan = np.clip(ranges, 0.0, self.max_range) / self.max_range

    # -------- 主計時器：先判斷是否「人工接管」，否則才跑模型 --------
    def on_timer(self):
        if self.last_scan is None:
            return

        now = time.time()

        # A) 是否在接管時間窗內？
        use_manual = (self.last_manual is not None) and (now - self.last_manual_time <= self.manual_to)

        # 計算距離最小值（實距離）
        d_min = float(np.min(self.last_scan) * self.max_range)

        if use_manual:
            # ---- 直接用手動（完全不管模型）----
            v = float(self.last_manual.linear.x)
            w = float(self.last_manual.angular.z)
            mode = 'MANUAL'
        else:
            # ---- 模型推論 ----
            x = torch.from_numpy(self.last_scan).to(self.device).unsqueeze(0)
            with torch.no_grad():
                y = self.model(x.float())
            v_hat = float(y[0,0].cpu().item())
            w_hat = float(y[0,1].cpu().item())

            v, w = v_hat, w_hat
            mode = 'IL'

            # 安全層
            if d_min < self.safety_stop:
                v = 0.0
                w = float(np.clip(w * self.slowturn_gain, -self.w_max, self.w_max))
                self.get_logger().warn(f'[safety] slow-turn: d_min={d_min:.2f} < {self.safety_stop:.2f}')
            else:
                # 增益
                v *= self.v_gain
                w *= self.w_gain
                # 最小地板（與牆距離 > 安全門檻 + 0.10 才套用，避免貼牆還硬拉）
                if d_min >= (self.safety_stop + 0.10):
                    if 0.0 < abs(v) < self.v_min:
                        v = self.v_min if v >= 0.0 else -self.v_min
                    if 0.0 < abs(w) < self.w_min:
                        w = self.w_min if w >= 0.0 else -self.w_min

        # 終端限幅
        v = float(np.clip(v, -self.v_max, self.v_max))
        w = float(np.clip(w, -self.w_max, self.w_max))

        # 寄出 Twist
        t = Twist()
        t.linear.x  = v
        t.angular.z = w
        self.pub_cmd.publish(t)

        # 節流 log + CSV
        if now - self._last_log_t >= (1.0 / max(self.log_rate, 1e-6)):
            self._last_log_t = now
            if use_manual:
                self.get_logger().info(f"mode={mode}  d_min={d_min:.3f}  →  v={v:.3f} w={w:.3f}")
                v_hat = w_hat = math.nan
            else:
                self.get_logger().info(f"mode={mode}  d_min={d_min:.3f}  v̂={v_hat:.3f} ŵ={w_hat:.3f}  →  v={v:.3f} w={w:.3f}")

            if self._csv_path:
                with open(self._csv_path, 'a', newline='') as f:
                    wcsv = csv.writer(f)
                    stamp_ns = self.get_clock().now().nanoseconds
                    wcsv.writerow([
                        stamp_ns, f"{d_min:.4f}", mode,
                        ("" if math.isnan(v_hat) else f"{v_hat:.4f}"),
                        ("" if math.isnan(w_hat) else f"{w_hat:.4f}"),
                        f"{v:.4f}", f"{w:.4f}"
                    ])

def main():
    rclpy.init()
    node = LidarILNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
