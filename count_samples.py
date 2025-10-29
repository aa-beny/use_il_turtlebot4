import numpy as np
import glob
import os

# --- 設定 ---
data_folder = "/home/open/ws_navlearn/data3"  # 【改成你的 data3 資料夾路徑】
recording_hz = 10.0                          # 【改成你錄製時用的 hz】
# --- 設定結束 ---

total_samples = 0
file_count = 0
possible_keys = ["cmd", "actions", "cmd_vel", "vel", "scan", "scans", "ranges"] # 可能的 key

# 找出資料夾裡所有的 .npz 檔案
files = glob.glob(os.path.join(data_folder, "*.npz"))

if not files:
    print(f"錯誤：在 {data_folder} 找不到任何 .npz 檔案。")
else:
    for f_path in files:
        file_count += 1
        try:
            data = np.load(f_path, allow_pickle=True)

            # 試著找一個存在的 key 來計算樣本數
            key_found = None
            for key in possible_keys:
                if key in data:
                    key_found = key
                    break

            if key_found:
                # 取得該 key 對應 array 的第一個維度 (樣本數)
                num_samples_in_file = data[key_found].shape[0]
                total_samples += num_samples_in_file
            else:
                print(f"警告：檔案 {os.path.basename(f_path)} 找不到可用的 key 來計算樣本數。")

        except Exception as e:
            print(f"錯誤：無法讀取或處理檔案 {os.path.basename(f_path)}: {e}")

    # 計算總秒數
    total_seconds = total_samples / recording_hz if recording_hz > 0 else 0

    print(f"--- 資料統計 ({data_folder}) ---")
    print(f"檔案數量: {file_count}")
    print(f"總樣本數: {total_samples}")
    print(f"錄製頻率: {recording_hz} Hz")
    print(f"估計總秒數: {total_seconds:.2f} 秒")
    print(f"估計總分鐘數: {total_seconds / 60:.2f} 分鐘")
