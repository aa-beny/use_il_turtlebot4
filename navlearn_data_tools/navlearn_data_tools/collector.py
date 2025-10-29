import rclpy, numpy as np, time, os
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

def resample_scan(ranges, angle_min, angle_max, bins=360):
    N = len(ranges)
    src = np.linspace(angle_min, angle_max, N, dtype=np.float32)
    dst = np.linspace(angle_min, angle_max, bins, dtype=np.float32)
    return np.interp(dst, src, ranges).astype(np.float32)

class Collector(Node):
    def __init__(self):
        super().__init__('il_collector')
        self.declare_parameter('save_dir', os.path.expanduser('~/ws_navlearn/data'))
        self.declare_parameter('bins', 360)
        self.declare_parameter('max_range', 3.5)
        self.save_dir = self.get_parameter('save_dir').get_parameter_value().string_value
        self.bins = self.get_parameter('bins').get_parameter_value().integer_value
        self.max_range = float(self.get_parameter('max_range').get_parameter_value().double_value)
        os.makedirs(self.save_dir, exist_ok=True)

        self.scan = None
        self.cmd = np.zeros(2, dtype=np.float32)  # (v, w)
        self.last_save = 0.0
        self.dt = 0.1

        self.create_subscription(LaserScan, '/scan', self.cb_scan, 10)
        self.create_subscription(Twist, '/cmd_vel', self.cb_cmd, 10)
        self.timer = self.create_timer(0.02, self.tick)  # 50 Hz 檢查是否該存一筆

        ts = time.strftime('%Y%m%d_%H%M%S')
        self.path = os.path.join(self.save_dir, f'session_{ts}.npz')
        self.buf_scan, self.buf_cmd = [], []
        self.get_logger().info(f'Start collecting → {self.path}')

    def cb_scan(self, msg: LaserScan):
        rng = np.array(msg.ranges, dtype=np.float32)
        rng = np.nan_to_num(rng, nan=msg.range_max, posinf=msg.range_max, neginf=0.0)
        rng = np.clip(rng, 0.0, self.max_range)
        rng = resample_scan(rng, msg.angle_min, msg.angle_max, self.bins)
        rng = rng / self.max_range  # → [0,1]
        self.scan = rng

    def cb_cmd(self, msg: Twist):
        self.cmd[0] = np.float32(msg.linear.x)
        self.cmd[1] = np.float32(msg.angular.z)

    def tick(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.scan is None: return
        if now - self.last_save < self.dt: return
        self.last_save = now
        self.buf_scan.append(self.scan.copy())
        self.buf_cmd.append(self.cmd.copy())
        if len(self.buf_scan) % 50 == 0:
            self.get_logger().info(f'collected {len(self.buf_scan)} samples')

    def destroy_node(self):
        if self.buf_scan:
            np.savez_compressed(self.path, scan=np.stack(self.buf_scan), cmd=np.stack(self.buf_cmd))
            self.get_logger().info(f'saved {len(self.buf_scan)} to {self.path}')
        super().destroy_node()

def main():
    rclpy.init()
    node = Collector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
if __name__ == '__main__': main()

