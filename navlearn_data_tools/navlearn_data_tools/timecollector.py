# navlearn_data_tools/collector_safe.py
# 作用：模仿學習資料收集（/scan -> binned 節距；/cmd_vel -> (v,w)）
#      增強：偵測最近距離 d_min，一旦小於 d_stop，就回溯丟掉最近 backtrack_sec 的樣本，存檔並結束。
# 說明：參數都可從 --ros-args -p ... 覆蓋；每行有中文註解，照著看就懂。

import os, time, math, collections
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class ILCollectorSafe(Node):
    def __init__(self):
        super().__init__('il_collector')

        # ========= 參數 =========
        self.save_dir    = self.declare_parameter('save_dir', os.path.expanduser('~/ws_navlearn/data')).value
        self.bins        = int(self.declare_parameter('bins', 360).value)          # 將雷射重採樣到幾個bin
        self.max_range   = float(self.declare_parameter('max_range', 3.5).value)   # 距離上限(裁切/正規化)
        self.hz          = float(self.declare_parameter('hz', 10.0).value)         # 取樣頻率
        # 安全相關
        self.d_stop      = float(self.declare_parameter('d_stop', 0.18).value)     # 觸發停錄門檻
        self.d_resume    = float(self.declare_parameter('d_resume', 0.24).value)   # 安全恢復（這版不再恢復，直接存檔離開；留給你調整）
        self.backtrack_s = float(self.declare_parameter('backtrack_sec', 2.0).value) # 回溯秒數

        os.makedirs(self.save_dir, exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S')
        self.out_path = os.path.join(self.save_dir, f'session_{ts}.npz')

        # ========= 緩衝與狀態 =========
        self.dt     = 1.0 / self.hz
        self.tail_n = max(1, int(self.backtrack_s * self.hz))  # 回溯樣本數
        self.scan_vec = None    # 最新的 binned scan
        self.vw       = np.zeros(2, dtype=np.float32)  # 最新 (v,w)

        # 使用 deque 做「可回溯」緩衝
        self.buf_scan = collections.deque()
        self.buf_cmd  = collections.deque()
        self.buf_dmin = collections.deque()

        # ========= 訂閱 =========
        self.create_subscription(LaserScan, '/scan', self.cb_scan, 10)
        self.create_subscription(Twist,     '/cmd_vel', self.cb_cmd,  10)

        # ========= 取樣 Timer =========
        self.timer = self.create_timer(self.dt, self.cb_tick)

        self.samples = 0
        self.last_log = time.time()
        self.get_logger().info(f"[il_collector]: Start collecting → {self.out_path}")

    # ---- 將 LaserScan 重採樣到固定 bins（簡易平均池化）----
    def bin_scan(self, ranges, angle_min, angle_max):
        arr = np.array(ranges, dtype=np.float32)
        # 去掉 nan/inf，裁切到 [0, max_range]
        arr[np.isnan(arr)] = self.max_range
        arr[np.isinf(arr)] = self.max_range
        arr = np.clip(arr, 0.0, self.max_range)

        m = len(arr)
        if m == self.bins:
            binned = arr
        else:
            # 將 m 點壓成 bins 點：把索引線性對應過去，再做區間平均
            idxs = (np.linspace(0, m, num=self.bins+1)).astype(int)
            binned = np.zeros(self.bins, dtype=np.float32)
            for i in range(self.bins):
                a, b = idxs[i], idxs[i+1]
                if b <= a: b = a+1
                binned[i] = np.mean(arr[a:b])
        # 正規化到 [0,1]
        binned = binned / self.max_range
        return binned

    # ---- /scan callback ----
    def cb_scan(self, msg: LaserScan):
        vec = self.bin_scan(msg.ranges, msg.angle_min, msg.angle_max)
        self.scan_vec = vec

    # ---- /cmd_vel callback ----
    def cb_cmd(self, msg: Twist):
        self.vw = np.array([msg.linear.x, msg.angular.z], dtype=np.float32)

    # ---- 每 dt 取一筆資料 ----
    def cb_tick(self):
        if self.scan_vec is None:
            return  # 還沒收到雷射

        # 反正 binned 是 [0,1]，拿實際距離方便理解
        dmin = float(np.min(self.scan_vec) * self.max_range)

        # 推進緩衝
        self.buf_scan.append(self.scan_vec.copy())
        self.buf_cmd.append(self.vw.copy())
        self.buf_dmin.append(dmin)
        self.samples += 1

        # 只保證「可回溯」，沒必要限制上限；想節省記憶體可加上限再寫檔分段
        now = time.time()
        if now - self.last_log > 5.0:
            self.get_logger().info(f"[il_collector]: collected {self.samples} samples (d_min={dmin:.3f} m)")
            self.last_log = now

        # ==== 安全判斷：一旦 d_min < d_stop → 丟尾巴，存檔並結束 ====
        if dmin < self.d_stop:
            # 回溯丟樣：把最近 tail_n 筆拿掉
            n_drop = min(self.tail_n, self.samples)
            for _ in range(n_drop):
                self.buf_scan.pop()
                self.buf_cmd.pop()
                self.buf_dmin.pop()
            self.samples -= n_drop

            self.get_logger().warn(
                f"[il_collector]: d_min={dmin:.3f} < d_stop={self.d_stop:.3f} → drop last {n_drop} and save & exit")

            self._save_and_exit()

    # ---- 存檔並乾淨退出 ----
    def _save_and_exit(self):
        scans = np.stack(self.buf_scan, axis=0) if len(self.buf_scan) else np.zeros((0, self.bins), dtype=np.float32)
        cmds  = np.stack(self.buf_cmd,  axis=0) if len(self.buf_cmd)  else np.zeros((0, 2), dtype=np.float32)
        dmins = np.array(self.buf_dmin, dtype=np.float32)

        np.savez_compressed(self.out_path, scan=scans, cmd=cmds, dmin=dmins,
                            bins=self.bins, max_range=self.max_range, hz=self.hz,
                            d_stop=self.d_stop, backtrack_sec=self.backtrack_s)
        self.get_logger().info(f"[il_collector]: saved {len(scans)} to {self.out_path}")

        # 優雅關閉
        self.destroy_timer(self.timer)
        self.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass

def main():
    rclpy.init()
    node = ILCollectorSafe()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # 手動 Ctrl-C：也把目前緩衝寫檔（不丟回溯）
        node._save_and_exit()

