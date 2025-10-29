# -*- coding: utf-8 -*-
# file: ~/ws_navlearn/train_il_lidar.py
import argparse, os, glob, math, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
# -*- coding: utf-8 -*-
# file: ~/ws_navlearn/train_il_lidar.py
import argparse, os, glob, math, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt  # <---【新增】匯入畫圖工具
# ----------------------------
# Dataset
# ----------------------------
POSSIBLE_SCAN_KEYS = ["scan", "scans", "ranges"]
POSSIBLE_CMD_KEYS  = ["cmd", "actions", "cmd_vel", "vel"]

def load_npz_robust(path):
    data = np.load(path, allow_pickle=True)
    scan_key = next((k for k in POSSIBLE_SCAN_KEYS if k in data), None)
    cmd_key  = next((k for k in POSSIBLE_CMD_KEYS  if k in data), None)
    if scan_key is None or cmd_key is None:
        raise KeyError("keys not found")
    scan = data[scan_key].astype(np.float32)
    cmd  = data[cmd_key].astype(np.float32)
    # cmd 可能是 (N,2) 或 dict，這裡只保 v=linear.x, w=angular.z
    if cmd.ndim == 1 and cmd.shape[0] == 2:
        cmd = cmd[None, :]
    assert scan.shape[0] == cmd.shape[0], f"length mismatch: {scan.shape} vs {cmd.shape}"
    return scan, cmd

class LidarILDataset(Dataset):
    def __init__(self, files, max_range=3.5, bins=360, augment=True):
        # X=題目(LiDAR), Y=答案(v, w)
        self.X, self.Y = [], []
        self.max_range = float(max_range)
        self.bins = int(bins)
        self.augment = augment
        kept, skipped = 0, 0
        # 一個一個 .npz 檔案打開
        for f in files:
            try:
                # load_npz_robust 去開檔案
                scan, cmd = load_npz_robust(f)
            except Exception:
                skipped += 1
                continue
            # 容忍不同 bins：若不一致就線性重採樣
            if scan.shape[1] != self.bins:
                x_src = np.linspace(0., 1., scan.shape[1], dtype=np.float32)
                x_dst = np.linspace(0., 1., self.bins, dtype=np.float32)
                # (用 np.interp 把它們全部變成 360)
                scan = np.interp(x_dst, x_src, scan).astype(np.float32)
            # 範圍正規化（錄製時已除以 max_range 的話這行等於無事）
            scan = np.clip(scan, 0.0, self.max_range) / self.max_range
            # 只取 (v, w) 兩欄
            cmd = cmd[:, :2].astype(np.float32)
            # 過濾停車全零的長段（保持一點點，用於學會停止）
            nz = np.any(np.abs(cmd) > 1e-6, axis=1)
            if nz.sum() < 10:
                skipped += 1
                continue
            self.X.append(scan)
            self.Y.append(cmd)
            kept += 1
        if kept == 0:
            raise RuntimeError("No valid samples found. Check your .npz keys and folder.")
        # 把所有 X (題目) Y (答案)疊起來
        self.X = np.vstack(self.X)
        self.Y = np.vstack(self.Y)
        # 打亂
        idx = np.arange(len(self.X))
        # 把所有資料「洗牌」
        np.random.shuffle(idx)
        self.X, self.Y = self.X[idx], self.Y[idx]
        print(f"[DATA] kept {kept} files, skipped {skipped} files → samples: {len(self.X)}")

    def __len__(self):
        return len(self.X)

    def _augment(self, x, y):
        # 左右鏡像（雷射反向 + 角速度取負）
        if random.random() < 0.5:
            x = x[::-1].copy()
            y = y.copy()
            y[1] = -y[1]
        # 角度小抖動（相位循環位移）
        if random.random() < 0.5:
            shift = random.randint(-5, 5)
            x = np.roll(x, shift)
        return x, y
    # 發考卷
    def __getitem__(self, i):
        # 抽第 i 張題目答案
        x = self.X[i]
        y = self.Y[i]
        # 如果「資料增強」開關是打開的
        if self.augment:
            x, y = self._augment(x, y)
        return torch.from_numpy(x), torch.from_numpy(y)

# ----------------------------
# Model（簡單三層 MLP）
# ----------------------------
class MLP(nn.Module):
    def __init__(self, bins=360):
        super().__init__()
        self.net = nn.Sequential(
            # 它看著 360 個雷射點，學會了一些「基本事實」（特徵）。
            # 例如，它可能學會了：
            # 特徵 1：「左前方 1 公尺內有東西」
            # 特徵 2：「正前方 3 公尺全空」
            # 特徵 3：「右邊是一整面牆」
            # ...總共學會了 256 個這樣的「單字」。
            nn.Linear(bins, 256),
            # 把負數的訊號過濾掉，只保留正數訊號，讓大腦學習更有效率。
            nn.ReLU(),
            # 把第一層的 256 個訊號，再做一次複雜的計算，變成新的 256 個訊號。
            # 有了這一層，模型就可以把這些「單字」組合起來，學會更複雜的「情境」或「策略」。
            # 它可以學會：
            # 策略 1 (走廊)：如果「特徵 1 (左有牆)」和「特徵 3 (右有牆)」和「特徵 2 (前全空)」同時發生 ➡️ 這代表「我正在一個走廊裡」，那我應該「保持直走」。
            # 策略 2 (左轉)：如果「特徵 1 (左前方有東西)」和「特徵 3 (右邊是空的)」同時發生 ➡️ 這代表「我該準備左轉了」。
            # 策略 3 (死路)：如果「特徵 1」和「特徵 2」和「特徵 3」... 幾乎全部都是「有東西」 ➡️ 這代表「我遇到死路了」，我應該「馬上停下並後退」。
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # (v, w)
        )
    def forward(self, x):
        return self.net(x)

# ----------------------------
# Train
# ----------------------------
def train(args):
    # rgs.data_dir 這個變數預設值是 ~/ws_navlearn/data 
    # os.path.expanduser args.data_dir把 ~ 符號展開成你電腦的「家目錄」完整路徑。
    # os.path.join( ... , "*.npz")所有以 .npz 結尾的檔案
    # glob.glob( ... )找出所有符合這個描述的檔案，然後回傳一個「清單 (list)」
    # sorted( ... )排好隊
    files = sorted(glob.glob(os.path.join(os.path.expanduser(args.data_dir), "*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npz in {args.data_dir}")
    # 把你設定的 bins=360 傳進去了。現在我們跳到 LidarILDataset
    ds = LidarILDataset(files, max_range=args.max_range, bins=args.bins, augment=not args.no_aug)

    val_ratio = float(args.val_ratio)
    n_val = max(1, int(len(ds) * val_ratio))
    n_train = len(ds) - n_val
    ds_train, ds_val = random_split(ds, [n_train, n_val])

    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True)
    dl_val   = DataLoader(ds_val,   batch_size=args.batch, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(bins=args.bins).to(device)
    # orch.optim 是 PyTorch 的「學習工具箱」。 Adam自動調參 lr 就是 Learning Rate (學習率)。args.lr 的值是 1e-3 也就是 0.001是一個公認最好用的「預設值」
    #     PyTorch 的兩大核心功能
    # 想像 PyTorch 是一個超級跑車：

    # GPU 加速 (你說的)：這是跑車的「V12 引擎」。它提供強大的動力，讓車子跑得飛快。

    # 自動微分 (Autograd)：這是跑車的「自動駕駛/導航系統」。它知道怎麼去「目的地」（也就是 loss 最小的地方）。
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 對 w 給較大權重（避免只學會直走）
    # 從「設定選單」讀取那個  的權重。
    lambda_w = args.lambda_w
    # 均方誤差
    mse = nn.MSELoss(reduction='mean')
    # 紀錄模擬考的「最好成績」inf 就是「無限大 (Infinity)」。在開始訓練前，你先把「最好成績」設定成無限大。
    # 這樣，你第一次考模擬考（第一個 epoch）得到的任何成績（例如 val_loss = 5.0），都一定會比「無限大」還要好，所以 5.0 就會立刻成為「最好成績」。
    best_val = math.inf
    # 紀錄「耐心值」
    patience = args.patience
    # 紀錄「退步次數」
    bad = 0

    train_losses = []  # <---【新增】存 train loss 歷史
    val_losses = []    # <---【新增】存 val loss 歷史


    for epoch in range(1, args.epochs + 1):
        # 訓練迴圈 (做練習本)
        model.train()
        tr_loss = 0.0
        for xb, yb in dl_train:
            # 考卷放上顯卡
            xb, yb = xb.to(device), yb.to(device)
            # 學生(model) 看了題目(xb)，寫出它的答案(pred)
            pred = model(xb)
            # MSE(v) + lambda_w * MSE(w)
            loss = mse(pred[:, 0], yb[:, 0]) + lambda_w * mse(pred[:, 1], yb[:, 1])
            # 學習前，先把舊筆記清空
            opt.zero_grad()
            # 老師(loss)「反向傳播」告訴學生錯在哪
            loss.backward()
            # 學生(opt)「更新大腦」
            opt.step()
            # loss.backward() 和 opt.step() 是 PyTorch 的魔法。你只要算出 loss（差距），backward 會自動算出大腦中每個參數該如何調整，step 會執行調整。
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(dl_train.dataset)

        train_losses.append(tr_loss) # 把這輪算出的 train loss 記錄下來

        # 驗證迴圈 (考模擬考)
        # model.eval()：告訴學生「現在是考試！不准學習，只要作答！」。
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for xb, yb in dl_val:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = mse(pred[:, 0], yb[:, 0]) + lambda_w * mse(pred[:, 1], yb[:, 1])
                val_loss += loss.item() * xb.size(0)
            val_loss /= len(dl_val.dataset)

            val_losses.append(val_loss)   # 把這輪算出的 val loss 記錄下來

        # 儲存和早停 (發獎狀或退學) 
        print(f"[E{epoch:03d}] train={tr_loss:.5f}  val={val_loss:.5f}")

        if val_loss + 1e-6 < best_val:
            best_val = val_loss# 刷新最佳成績 歷史上出現過的「最低
            bad = 0# 重設「退步次數」
            os.makedirs(os.path.dirname(os.path.expanduser(args.out_path)), exist_ok=True)
            torch.save({"model": model.state_dict(),
                        "bins": args.bins, "max_range": args.max_range}, os.path.expanduser(args.out_path))
            print(f"  ↳ saved to {args.out_path}")
        else:
            bad += 1 # 退步了
            if bad >= patience:
                print("  ↳ early stop")# 老師沒耐心了，提早下課！
                break# 跳出 for epoch 迴圈
        # 這就是早停 (Early Stopping)。如果學生在模擬考 (val_loss) 上，連續 patience (例如 10) 次都沒有進步，我們就當作它已經學到底了，直接停止訓練。這樣可以節省時間，也避免它開始『死背』練習本的答案（Overfitting）
    print("best val:", best_val)
    # ==========================================================
    #            【新增】整個畫圖區塊 開始
    # ==========================================================
    epochs_run = len(train_losses) # 看看實際跑了幾個 epoch
    if epochs_run > 0: # 確保至少有跑一輪
        plt.figure(figsize=(10, 6)) # 設定圖片大小
        # 畫 training loss 曲線
        plt.plot(range(1, epochs_run + 1), train_losses, marker='o', linestyle='-', label='Training Loss')
        # 畫 validation loss 曲線
        plt.plot(range(1, epochs_run + 1), val_losses, marker='x', linestyle='--', label='Validation Loss')

        plt.xlabel('Epoch') # x 軸標籤
        plt.ylabel('Loss (MSE)') # y 軸標籤
        plt.title('Training and Validation Loss Curve') # 圖表標題
        plt.legend() # 顯示圖例 (右上角告訴你哪條線是什麼)
        plt.grid(True) # 顯示格線

        # 找出最低 validation loss 的點，並標示出來
        best_epoch = np.argmin(val_losses) + 1
        plt.scatter(best_epoch, best_val, color='red', s=100, zorder=5, label=f'Best Val Loss ({best_val:.4f}) at Epoch {best_epoch}')
        plt.legend() # 更新圖例

        # 決定圖片儲存路徑 (跟模型檔名關聯)
        plot_path = os.path.splitext(os.path.expanduser(args.out_path))[0] + "_loss.png"
        try:
            plt.savefig(plot_path) # 儲存圖片
            print(f"Loss curve saved to: {plot_path}")
        except Exception as e:
            print(f"Warning: Failed to save loss curve plot: {e}")

        # 如果你想讓圖片直接跳出來，可以取消下面這行的註解
        # plt.show()
    else:
        print("No epochs were run, skipping plot generation.")
    # ==========================================================
    #            【新增】整個畫圖區塊 結束
    # ==========================================================

# ----------------------------
# Args
# ----------------------------
if __name__ == "__main__":
    # 定義了所有你可以從外面調整的開關。
    ap = argparse.ArgumentParser()
    # 你放 .npz 資料的資料夾
    ap.add_argument("--data_dir",   type=str, default="~/ws_navlearn/data")
    # 模型要存在哪裡。
    ap.add_argument("--out_path",   type=str, default="~/ws_navlearn/models/il_lidar.pt")
    # 要把 LiDAR 360 度的資料切成幾份
    ap.add_argument("--bins",       type=int, default=360)
    ap.add_argument("--max_range",  type=float, default=3.5)
    ap.add_argument("--epochs",     type=int, default=200)
    ap.add_argument("--batch",      type=int, default=512)
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--val_ratio",  type=float, default=0.1)
    ap.add_argument("--patience",   type=int, default=10)
    # 懲罰權重
    ap.add_argument("--lambda_w",   type=float, default=3.0)
    # 要不要做「資料增強」
    ap.add_argument("--no_aug",     action="store_true")
    # 設定讀取完這包設定 (args)，去呼叫 train 函數
    args = ap.parse_args()
    train(args)

