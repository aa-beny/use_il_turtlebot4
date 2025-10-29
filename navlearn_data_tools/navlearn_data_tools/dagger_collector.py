# -*- coding: utf-8 -*-
# DAgger collector (manual-on-demand, robust trigger)
# 只在 /cmd_vel/manual 有「真的非零手動訊號」時，才開始錄。
# 選配：require_arm_for_manual=true -> 必須先靠近( d_min < arm_thresh ) 才接受手動觸發。
import os, time, math, collections
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class RingBuf:
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.q = collections.deque(maxlen=maxlen)
    def append(self, x):
        self.q.append(x)
    def clear(self):
        self.q.clear()
    def __len__(self):
        return len(self.q)
    def to_np(self):
        if not self.q: return None
        return np.stack(self.q, axis=0)

class DaggerCollector(Node):
    def __init__(self):
        super().__init__('dagger_collector')
        p = self.declare_and_get

        self.save_dir   = p('save_dir', str, '/tmp/data_dagger')
        self.topic_scan = p('topic_scan', str, '/scan')
        # **只看 manual**：你真的按鍵才會來這條 topic
        self.topic_manual = p('topic_cmd_manual', str, '/cmd_vel/manual')

        # 幀率 & LIDAR
        self.hz        = p('hz', float, 10.0)
        self.bins      = p('bins', int, 360)
        self.max_range = p('max_range', float, 3.5)

        # 觸發/收尾/冷卻
        self.trig_eps   = p('trig_eps', float, 0.20)   # 手動向量長度門檻（建議 0.2~0.3）
        self.trig_hold  = p('trig_hold', int, 2)       # 連續幾個取樣 > trig_eps
        self.manual_quiet = p('manual_quiet', float, 1.0)  # 放手多久才收尾
        self.refrac_sec = p('refrac_sec', float, 4.0)      # 冷卻期
        self.max_seg_sec = p('max_seg_sec', float, 8.0)    # 最長片段

        # 近距 arming
        self.arm_thresh  = p('arm_thresh', float, 0.45)
        self.stop_thresh = p('stop_thresh', float, 0.0)     # 這版預設關閉自動靠太近就錄
        self.require_arm_for_manual = p('require_arm_for_manual', bool, True)

        self.pre_sec  = p('pre_sec', float, 2.0)
        self.post_sec = p('post_sec', float, 1.0)

        os.makedirs(self.save_dir, exist_ok=True)

        # 狀態
        self.state = 'IDLE'   # IDLE -> ARMED -> RECORDING -> COOLDOWN
        self.last_save_wall = 0.0
        self.last_manual_time = 0.0
        self.last_nonzero_time = 0.0
        self.manual_mag_hist = collections.deque(maxlen=self.trig_hold)
        self.record_start_wall = None

        # 緩衝
        self.buf_scan = RingBuf(maxlen=int((self.pre_sec+self.max_seg_sec+self.post_sec)*self.hz))
        self.buf_cmd  = RingBuf(maxlen=int((self.pre_sec+self.max_seg_sec+self.post_sec)*self.hz))
        self.buf_dmin = RingBuf(maxlen=int((self.pre_sec+self.max_seg_sec+self.post_sec)*self.hz))

        # 目前值
        self.cur_scan = None
        self.cur_cmd  = np.zeros(2, np.float32)
        self.cur_dmin = 9e9

        qos = QoSProfile(depth=1)
        self.create_subscription(LaserScan, self.topic_scan, self.on_scan, qos)
        self.create_subscription(Twist, self.topic_manual, self.on_manual, qos)

        # 定時主回圈
        self.timer = self.create_timer(1.0/self.hz, self.on_timer)

        self.get_logger().info(
            f"DAgger collector up. Save to {self.save_dir}\n"
            f"scan: {self.topic_scan}  manual: {self.topic_manual}\n"
            f"arm_thresh={self.arm_thresh} stop_thresh={self.stop_thresh} trig_eps={self.trig_eps} trig_hold={self.trig_hold}\n"
            f"pre={self.pre_sec}s post={self.post_sec}s hz={self.hz} bins={self.bins} require_arm_for_manual={self.require_arm_for_manual}"
        )

    # 參數小工具
    # 參數小工具
    # 參數小工具
    def declare_and_get(self, name, typ, default):
        self.declare_parameter(name, default)
        val = self.get_parameter(name).get_parameter_value()
        # ---
        # 這是【終極修正版】
        # ---
        if typ == str:
            return val.string_value
        elif typ == float:
            return val.double_value   # <--- 修正點 (float -> double_value)
        elif typ == int:
            return val.integer_value  # <--- (順便修正 int)
        elif typ == bool:
            return val.bool_value     # <--- (順便修正 bool)
        else:
            # 其它類型（雖然我們這裡沒用到）
            return getattr(val, f'{typ.__name__}_value')

    # ---- Callbacks ----
    def on_scan(self, msg: LaserScan):
        # 轉成固定 bins，計算 d_min
        rng = np.array(msg.ranges, dtype=np.float32)
        rng = np.clip(rng, 0.0, self.max_range)
        if len(rng) != self.bins:
            x_src = np.linspace(0., 1., len(rng), dtype=np.float32)
            x_dst = np.linspace(0., 1., self.bins, dtype=np.float32)
            rng = np.interp(x_dst, x_src, rng).astype(np.float32)
        self.cur_scan = rng
        self.cur_dmin = float(np.min(rng))

    def on_manual(self, msg: Twist):
        v = float(msg.linear.x)
        w = float(msg.angular.z)
        self.cur_cmd[:] = (v, w)
        mag = math.hypot(v, w)
        now = time.time()
        self.last_manual_time = now
        if mag > 1e-6:
            self.last_nonzero_time = now
        self.manual_mag_hist.append(mag)

    # ---- 主回圈 ----
    # ---- 主回圈 ----
    def on_timer(self):
        now = time.time()
        # 緩衝
        if self.cur_scan is not None:
            self.buf_scan.append(self.cur_scan.copy())
            self.buf_cmd.append(self.cur_cmd.copy())
            self.buf_dmin.append(np.array([self.cur_dmin], np.float32))

        # 檢查手動觸發
        manual_triggered = self._manual_active()

        # 狀態機
        if self.state == 'IDLE':
            # 條件1: (在 "不必靠近" 模式下) 檢查手動觸發
            if (not self.require_arm_for_manual) and manual_triggered:
                self.state = 'RECORDING'
                self.record_start_wall = now
                self.get_logger().info("[trigger] manual (open area) → start record")
            # 條件2: (在 "必須靠近" 模式下) 檢查是否進入 "靠近" 狀態
            elif self.require_arm_for_manual and (self.cur_dmin < self.arm_thresh):
                self.state = 'ARMED'
                self.get_logger().info(f"[arming] d_min={self.cur_dmin:.3f} < {self.arm_thresh}")

        elif self.state == 'ARMED':
            # 檢查是否手動觸發 (在 "必須靠近" 模式下)
            if manual_triggered:
                self.state = 'RECORDING'
                self.record_start_wall = now
                self.get_logger().info("[trigger] manual (armed) → start record")
            # 檢查是否離開 "靠近" 狀態 (如果又開遠了)
            elif self.cur_dmin >= self.arm_thresh:
                self.state = 'IDLE'
                self.get_logger().info("[disarming] d_min is safe")

        elif self.state == 'RECORDING':
            # 收尾條件：放手夠久 or 時間過長
            too_long = (now - self.record_start_wall > self.max_seg_sec)
            quiet_enough = (now - self.last_nonzero_time) > self.manual_quiet
            if quiet_enough or too_long:
                self._save(now)
                self.state = 'COOLDOWN'
                self.last_save_wall = now

        elif self.state == 'COOLDOWN':
            if now - self.last_save_wall >= self.refrac_sec:
                # 回到初始
                self.state = 'IDLE'
                self.buf_scan.clear(); self.buf_cmd.clear(); self.buf_dmin.clear()

    # 手動是否真的在作用（連續多幀 > trig_eps 且最近 0.2s 有收到 manual）
    def _manual_active(self):
        recently = (time.time() - self.last_manual_time) < 0.2
        if not recently: 
            return False
        if len(self.manual_mag_hist) < self.manual_mag_hist.maxlen:
            return False
        return all(m > self.trig_eps for m in self.manual_mag_hist)

    def _arm_ok(self):
        if not self.require_arm_for_manual:
            return True
        return self.cur_dmin < self.arm_thresh

    def _save(self, now):
        # 取 pre/post
        N = len(self.buf_scan.q)
        if N == 0: 
            self.get_logger().warn("[save] buffer empty"); 
            return
        # 從尾巴往前找，截取 post_sec
        post_frames = int(self.post_sec * self.hz)
        pre_frames  = int(self.pre_sec  * self.hz)
        scan = self.buf_scan.to_np()
        cmd  = self.buf_cmd.to_np()
        dmin = self.buf_dmin.to_np().squeeze(-1)

        # 只取最後 (pre + 手動 + post) 這段 —— 簡化：整段都存
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
        path = os.path.join(self.save_dir, f"dagger_{ts}.npz")
        np.savez_compressed(path, scan=scan, cmd=cmd, dmin=dmin)
        dmin_min = float(np.min(dmin)) if dmin.size else float('nan')
        dmin_mean = float(np.mean(dmin)) if dmin.size else float('nan')
        self.get_logger().info(
            f"[save] {path}\n"
            f"      scan {scan.shape}  cmd {cmd.shape}  dmin[min/mean]={dmin_min:.3f}/{dmin_mean:.3f}"
        )

def main():
    rclpy.init()
    node = DaggerCollector()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
