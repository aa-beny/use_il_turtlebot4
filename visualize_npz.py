# -*- coding: utf-8 -*-
# file: ~/ws_navlearn/visualize_npz.py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# (跟 train_il_lidar.py 裡一樣的 robust loader)
POSSIBLE_SCAN_KEYS = ["scan", "scans", "ranges"]
POSSIBLE_CMD_KEYS  = ["cmd", "actions", "cmd_vel", "vel"]

def load_npz_robust(path):
    try:
        data = np.load(path, allow_pickle=True)
        scan_key = next((k for k in POSSIBLE_SCAN_KEYS if k in data), None)
        cmd_key  = next((k for k in POSSIBLE_CMD_KEYS  if k in data), None)
        if scan_key is None or cmd_key is None:
            raise KeyError(f"必要的 key ('scan' or 'cmd') 在 {path} 中找不到")
        scan = data[scan_key].astype(np.float32)
        cmd  = data[cmd_key].astype(np.float32)
        # 確保 cmd 是 (N, 2)
        if cmd.ndim == 1:
            if cmd.shape[0] == 2:
                cmd = cmd[None, :] # 如果只有一筆 (2,) -> (1, 2)
            else:
                 raise ValueError(f"Cmd 是一維但長度不是 2: {cmd.shape}")
        elif cmd.ndim == 2:
             if cmd.shape[1] < 2:
                 raise ValueError(f"Cmd 是二維但欄數少於 2: {cmd.shape}")
             cmd = cmd[:, :2] # 只取前兩欄 v, w
        else:
            raise ValueError(f"Cmd 維度不正確: {cmd.ndim}")

        if scan.ndim == 1:
             scan = scan[None, :] # (bins,) -> (1, bins)

        assert scan.shape[0] == cmd.shape[0], \
               f"Scan 和 Cmd 樣本數不匹配 in {path}: {scan.shape[0]} vs {cmd.shape[0]}"
        return scan, cmd
    except Exception as e:
        print(f"錯誤：讀取檔案 {path} 失敗: {e}")
        return None, None

def visualize_sample(scan_data, cmd_data, index, filename, total_samples):
    """
    視覺化單一筆 scan 和 cmd 資料
    scan_data: (N, bins) 的 scan 陣列
    cmd_data: (N, 2) 的 cmd 陣列
    index: 要顯示的樣本索引 (0 到 N-1)
    filename: 原始 .npz 檔名 (用於標題)
    total_samples: 檔案中的總樣本數
    """
    if not (0 <= index < total_samples):
        print(f"錯誤：索引 {index} 超出範圍 (0 到 {total_samples-1})")
        return

    scan_single = scan_data[index] # (bins,)
    cmd_single = cmd_data[index]   # (2,)
    v = cmd_single[0]
    w = cmd_single[1]
    bins = len(scan_single)

    # --- 繪製 LiDAR Scan (極座標圖) ---
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)

    # 產生角度 (假設 360 度掃描)
    angles = np.linspace(-np.pi, np.pi, bins, endpoint=False)

    # 繪製雷射點
    # 注意：scan_single 已經是正規化到 [0,1] 的，如果想看實際距離，需要乘以 max_range
    # 這裡我們先畫正規化的值
    ax.scatter(angles, scan_single, s=5, alpha=0.7, label='LiDAR Points (Normalized)')
    ax.plot(angles, scan_single, linewidth=0.5, alpha=0.5) # 可以加上線條連接

    # 設定極座標圖範圍和方向
    ax.set_theta_zero_location("N") # 0 度角在正北方 (x 軸)
    ax.set_theta_direction(-1)      # 角度順時針增加
    ax.set_ylim(0, 1.1)             # y 軸範圍 (正規化後最大是 1)
    ax.set_title(f"File: {filename}\nSample Index: {index}/{total_samples-1}\nCommand: v={v:.3f}, w={w:.3f}",
                 va='bottom')
    ax.grid(True)
    # ax.legend() # 點太多，圖例可能不好看

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="視覺化 NPZ 檔案中的 LiDAR scan 和 command 資料")
    parser.add_argument("npz_path", type=str, help="單一 .npz 檔案的路徑")
    parser.add_argument("-i", "--index", type=int, default=0, help="要顯示的樣本索引 (預設: 0)")
    args = parser.parse_args()

    if not os.path.exists(args.npz_path) or not args.npz_path.endswith('.npz'):
        print(f"錯誤：檔案不存在或不是 .npz 檔案: {args.npz_path}")
    else:
        scans, cmds = load_npz_robust(args.npz_path)
        if scans is not None and cmds is not None:
            num_samples = scans.shape[0]
            if num_samples > 0:
                 visualize_sample(scans, cmds, args.index, os.path.basename(args.npz_path), num_samples)
            else:
                 print(f"錯誤：檔案 {args.npz_path} 中沒有有效的樣本。")
