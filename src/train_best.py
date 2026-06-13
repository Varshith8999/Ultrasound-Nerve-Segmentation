"""
train_best.py  —  Best model for Ultrasound Nerve Segmentation
==============================================================

Architecture : ResUNet-CBAM-ASPP
  Encoder    : 4 × Residual blocks + Squeeze-and-Excitation  (64→128→256→512)
  Bridge     : Atrous Spatial Pyramid Pooling  (multi-scale context capture)
  Skip gates : CBAM (Channel + Spatial attention) on every skip connection
  Decoder    : 4 × Residual blocks + bilinear upsampling
  Supervision: Deep auxiliary outputs at dec1/dec2/dec3 (training only)

Training     : AdamW + OneCycleLR + AMP + gradient clipping
Augmentation : albumentations v2  (elastic, grid distortion, noise, brightness …)
Inference    : Batched prediction with 4-fold TTA
"""

# =========================
# IMPORTS
# =========================
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import matplotlib.pyplot as plt

# =========================
# SPEED / PRECISION SETTINGS
# =========================
torch.backends.cudnn.benchmark        = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.set_float32_matmul_precision("high")

# =========================
# PATHS  (same data as train_chat1.py)
# =========================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH   = os.path.join(os.path.dirname(_SCRIPT_DIR), "ultrasound-nerve-segmentation")
TRAIN_PATH  = os.path.join(BASE_PATH, "train")
TEST_PATH   = os.path.join(BASE_PATH, "test")
MODEL_PATH  = os.path.join(_SCRIPT_DIR, "best_model_v2.pth")

IMG_SIZE   = 256
IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}

print("TRAIN PATH:", TRAIN_PATH)
print("TEST  PATH:", TEST_PATH)

# =========================
# GPU
# =========================
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = torch.cuda.is_available()
print(f"Device : {device}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# =========================
# PREPROCESSING
# =========================
def apply_clahe(img: np.ndarray) -> np.ndarray:
    """CLAHE contrast enhancement — returns float32 in [0,1]."""
    u8    = (img * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return (clahe.apply(u8) / 255.0).astype(np.float32)

# =========================
# AUGMENTATION  (albumentations v2, applied in Dataset.__getitem__)
# =========================
_train_aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
    A.ElasticTransform(alpha=1, sigma=50, p=0.35),
    A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.25),
    A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.2),
    A.GaussNoise(p=0.2),
    A.CoarseDropout(
        num_holes_range=(1, 4),
        hole_height_range=(16, 48),
        hole_width_range=(16, 48),
        fill=0.0,
        p=0.2,
    ),
])

# =========================
# DATASET
# =========================
class UltrasoundDataset(Dataset):
    def __init__(self, images: np.ndarray, masks: np.ndarray, augment: bool = False):
        # Keep as numpy; convert to tensor in __getitem__ so augmentation is clean
        self.images  = images   # N×H×W×1  float32
        self.masks   = masks    # N×H×W×1  float32
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img  = self.images[idx].copy()   # H×W×1 float32
        mask = self.masks[idx].copy()    # H×W×1 float32

        if self.augment:
            # albumentations expects mask as H×W (no channel dim)
            result = _train_aug(image=img, mask=mask[..., 0])
            img    = result["image"]
            mask   = result["mask"][..., np.newaxis]

        # H×W×C → C×H×W
        img_t  = torch.from_numpy(img).permute(2, 0, 1)
        mask_t = torch.from_numpy(mask).permute(2, 0, 1)
        return img_t, mask_t

# =========================
# DATA LOADERS
# =========================
def load_data(min_nerve_pixels: int = 100):
    """
    Load and preprocess all training images.
    Skips empty masks (59 % of dataset) — they destroy the dice metric.
    """
    images, masks = [], []
    n_empty = n_bad = 0

    all_files   = os.listdir(TRAIN_PATH)
    image_files = [
        f for f in all_files
        if "_mask" not in f
        and os.path.splitext(f)[1].lower() in IMAGE_EXTS
    ]

    for img_file in image_files:
        ext       = os.path.splitext(img_file)[1]
        mask_file = img_file.replace(ext, f"_mask{ext}")
        img_path  = os.path.join(TRAIN_PATH, img_file)
        mask_path = os.path.join(TRAIN_PATH, mask_file)

        if not os.path.exists(mask_path):
            continue

        img  = cv2.imread(img_path,  cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            n_bad += 1
            continue
        if int(mask.sum()) < min_nerve_pixels:
            n_empty += 1
            continue

        img  = cv2.resize(img,  (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
        img  = apply_clahe(img)
        mask = (cv2.resize(mask, (IMG_SIZE, IMG_SIZE)) > 127).astype(np.float32)

        images.append(img[...,  np.newaxis])   # H×W×1
        masks.append(mask[..., np.newaxis])

    print(f"Loaded {len(images)} nerve-present images | "
          f"skipped {n_empty} empty, {n_bad} unreadable")

    X   = np.array(images, dtype=np.float32)
    y   = np.array(masks,  dtype=np.float32)
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def load_test():
    images, names = [], []
    for f in sorted(os.listdir(TEST_PATH)):
        if os.path.splitext(f)[1].lower() not in IMAGE_EXTS:
            continue
        img = cv2.imread(os.path.join(TEST_PATH, f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
        img = apply_clahe(img)
        images.append(img[..., np.newaxis])
        names.append(f)
    return np.array(images, dtype=np.float32), names

# =========================
# MODEL BUILDING BLOCKS
# =========================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation: recalibrates channel importance."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.se(x).view(x.size(0), x.size(1), 1, 1)
        return x * w


class ResBlock(nn.Module):
    """
    Pre-activation residual block with SE recalibration.
    Pre-activation (BN→ReLU→Conv) gives better gradient flow than post-activation.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        )
        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1, bias=False)
            if in_ch != out_ch else nn.Identity()
        )
        self.se   = SEBlock(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.se(self.conv(x)) + self.skip(x))


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Channel attention → Spatial attention applied sequentially.
    Focuses the decoder on the most relevant regions of each skip feature.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        # Channel attention (avg + max pooling paths)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
        )
        # Spatial attention (7×7 conv on avg/max pooled channel maps)
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        # Channel attention
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        mx  = F.adaptive_max_pool2d(x, 1).view(b, c)
        ca  = torch.sigmoid(self.fc(avg) + self.fc(mx)).view(b, c, 1, 1)
        x   = x * ca
        # Spatial attention
        avg_s = torch.mean(x, dim=1, keepdim=True)
        max_s = torch.max(x,  dim=1, keepdim=True)[0]
        sa    = self.spatial(torch.cat([avg_s, max_s], dim=1))
        return x * sa


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling.
    Captures multi-scale context at the bottleneck using dilated convolutions.
    At 16×16 spatial resolution, rates [1,2,4,6] give receptive fields
    that cover local details through the whole feature map.
    """
    def __init__(self, in_ch: int, out_ch: int, rates=(1, 2, 4, 6)):
        super().__init__()
        branch_ch = out_ch // (len(rates) + 1)   # even split across branches

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, branch_ch, 3,
                          padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(branch_ch),
                nn.ReLU(inplace=True),
            )
            for r in rates
        ])
        # Global average pooling branch captures image-level context
        self.global_br = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, branch_ch, 1, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True),
        )
        total_ch = branch_ch * (len(rates) + 1)
        self.project = nn.Sequential(
            nn.Conv2d(total_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    def forward(self, x):
        feats  = [b(x) for b in self.branches]
        global_feat = F.interpolate(
            self.global_br(x), size=x.shape[-2:],
            mode="bilinear", align_corners=True
        )
        feats.append(global_feat)
        return self.project(torch.cat(feats, dim=1))


# =========================
# FULL MODEL
# =========================
class ResUNetCBAMASPP(nn.Module):
    """
    ResUNet with CBAM skip gates, ASPP bridge, and deep supervision.

    During training  → returns (main_logits, aux1, aux2, aux3)
    During inference → returns  main_logits  only
    """
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # ---- Encoder ----
        self.enc1 = ResBlock(1,   64)    # 256×256 → 256×256
        self.enc2 = ResBlock(64,  128)   # 128×128
        self.enc3 = ResBlock(128, 256)   # 64×64
        self.enc4 = ResBlock(256, 512)   # 32×32

        # ---- Bridge ----
        self.aspp = ASPP(512, 512)       # 16×16  (after pool from enc4)

        # ---- CBAM gates on skip connections ----
        self.cbam4 = CBAM(512)
        self.cbam3 = CBAM(256)
        self.cbam2 = CBAM(128)
        self.cbam1 = CBAM(64)

        # ---- Decoder ----
        self.dec1 = ResBlock(512 + 512, 256)   # up(aspp) + cbam4
        self.dec2 = ResBlock(256 + 256, 128)   # up(dec1) + cbam3
        self.dec3 = ResBlock(128 + 128,  64)   # up(dec2) + cbam2
        self.dec4 = ResBlock( 64 +  64,  32)   # up(dec3) + cbam1

        # ---- Main output ----
        self.out  = nn.Conv2d(32, 1, 1)

        # ---- Deep supervision heads (training only) ----
        self.aux1 = nn.Conv2d(256, 1, 1)   # from dec1 (32×32)
        self.aux2 = nn.Conv2d(128, 1, 1)   # from dec2 (64×64)
        self.aux3 = nn.Conv2d( 64, 1, 1)   # from dec3 (128×128)

    def forward(self, x):
        h, w = x.shape[-2:]

        # Encoder
        c1 = self.enc1(x)
        c2 = self.enc2(self.pool(c1))
        c3 = self.enc3(self.pool(c2))
        c4 = self.enc4(self.pool(c3))

        # Bridge
        bridge = self.aspp(self.pool(c4))

        # Decoder with CBAM-gated skips
        d1 = self.dec1(torch.cat([self.up(bridge),  self.cbam4(c4)], dim=1))
        d2 = self.dec2(torch.cat([self.up(d1),      self.cbam3(c3)], dim=1))
        d3 = self.dec3(torch.cat([self.up(d2),      self.cbam2(c2)], dim=1))
        d4 = self.dec4(torch.cat([self.up(d3),      self.cbam1(c1)], dim=1))

        main = self.out(d4)

        if self.training:
            # Upsample auxiliary outputs to input resolution for loss computation
            a1 = F.interpolate(self.aux1(d1), size=(h, w), mode="bilinear", align_corners=True)
            a2 = F.interpolate(self.aux2(d2), size=(h, w), mode="bilinear", align_corners=True)
            a3 = F.interpolate(self.aux3(d3), size=(h, w), mode="bilinear", align_corners=True)
            return main, a1, a2, a3

        return main   # raw logits

# =========================
# LOSS FUNCTIONS
# =========================
def tversky_loss(y_true, logits, alpha=0.3, beta=0.7, smooth=1.0):
    """
    Tversky loss — alpha penalises FP, beta penalises FN.
    beta > alpha means the model is pushed to find every pixel of nerve
    rather than staying cautious.
    """
    p  = torch.sigmoid(logits)
    tp = (y_true * p).sum()
    fp = ((1 - y_true) * p).sum()
    fn = (y_true * (1 - p)).sum()
    return 1.0 - (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)


def focal_loss(y_true, logits, gamma=2.0, alpha=0.25):
    bce = F.binary_cross_entropy_with_logits(logits, y_true, reduction="none")
    pt  = torch.exp(-bce)
    return torch.mean(alpha * (1.0 - pt) ** gamma * bce)


def combined_loss(y_true, logits):
    return tversky_loss(y_true, logits) + focal_loss(y_true, logits)


def deep_supervised_loss(y_true, model_out):
    """
    Weighted sum over main + 3 auxiliary outputs.
    Auxiliary weights decay so earlier features contribute less.
    """
    if isinstance(model_out, tuple):
        main, a1, a2, a3 = model_out
        loss  = combined_loss(y_true, main)
        loss += 0.4 * combined_loss(y_true, a1)
        loss += 0.3 * combined_loss(y_true, a2)
        loss += 0.2 * combined_loss(y_true, a3)
        return loss, main
    return combined_loss(y_true, model_out), model_out

# =========================
# METRIC  — per-sample dice (accurate on imbalanced batches)
# =========================
def batch_dice(y_true, logits, smooth=1.0):
    p  = torch.sigmoid(logits)
    b  = y_true.size(0)
    yt = y_true.reshape(b, -1)
    yp = p.reshape(b, -1)
    inter = (yt * yp).sum(dim=1)
    return ((2.0 * inter + smooth) / (yt.sum(dim=1) + yp.sum(dim=1) + smooth)).mean()

# =========================
# TRAINING LOOP
# =========================
def train_model(model, train_loader, val_loader, epochs: int = 80):
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr          = 5e-4,
        steps_per_epoch = len(train_loader),
        epochs          = epochs,
        pct_start       = 0.1,
        anneal_strategy = "cos",
    )
    scaler   = torch.amp.GradScaler("cuda") if use_amp else None
    best_val = float("inf")
    patience = 0
    MAX_WAIT = 20

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        t_loss = t_dice = 0.0
        for imgs, msks in train_loader:
            imgs, msks = (imgs.to(device, non_blocking=True),
                          msks.to(device, non_blocking=True))
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    out        = model(imgs)
                    loss, main = deep_supervised_loss(msks, out)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                out        = model(imgs)
                loss, main = deep_supervised_loss(msks, out)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            t_loss += loss.item()
            t_dice += batch_dice(msks, main).item()

        # ---- Validate ----
        model.eval()
        v_loss = v_dice = 0.0
        with torch.no_grad():
            for imgs, msks in val_loader:
                imgs, msks = (imgs.to(device, non_blocking=True),
                              msks.to(device, non_blocking=True))
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        logits = model(imgs)
                else:
                    logits = model(imgs)
                v_loss += combined_loss(msks, logits).item()
                v_dice += batch_dice(msks, logits).item()

        n_tr  = len(train_loader); n_vl = len(val_loader)
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{epochs} | LR {lr_now:.2e} | "
            f"Train loss {t_loss/n_tr:.4f}  dice {t_dice/n_tr:.4f} | "
            f"Val   loss {v_loss/n_vl:.4f}  dice {v_dice/n_vl:.4f}"
        )

        if v_loss < best_val:
            best_val = v_loss
            patience = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  -> checkpoint saved (epoch {epoch})")
        else:
            patience += 1
            if patience >= MAX_WAIT:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    return model

# =========================
# TEST-TIME AUGMENTATION  (4-fold)
# =========================
def _predict_once(model, batch: torch.Tensor) -> torch.Tensor:
    """Single forward pass → probabilities."""
    if use_amp:
        with torch.amp.autocast("cuda"):
            logits = model(batch)
    else:
        logits = model(batch)
    return torch.sigmoid(logits)


def predict_tta(model, batch: torch.Tensor) -> torch.Tensor:
    """
    Average 4 orientations: original, h-flip, v-flip, 180° rotation.
    Each prediction is un-flipped before averaging so they align spatially.
    """
    with torch.no_grad():
        p  = _predict_once(model, batch)
        p += _predict_once(model, torch.flip(batch, [-1])).flip([-1])
        p += _predict_once(model, torch.flip(batch, [-2])).flip([-2])
        p += _predict_once(model, torch.rot90(batch, 2, [-2, -1])).rot90(-2, [-2, -1])
    return p / 4.0


def predict_batched(model, X_np: np.ndarray, batch_size: int = 32) -> np.ndarray:
    """Run TTA prediction in mini-batches to avoid VRAM overflow."""
    model.eval()
    all_preds = []
    total = len(X_np)
    for start in range(0, total, batch_size):
        end   = min(start + batch_size, total)
        batch = (torch.from_numpy(X_np[start:end])
                 .permute(0, 3, 1, 2)
                 .to(device, non_blocking=True))
        p = predict_tta(model, batch)
        all_preds.append(p.cpu().numpy())
        print(f"  predicted {end}/{total}", end="\r")
    print()
    return np.concatenate(all_preds, axis=0)   # N×1×H×W

# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    # ---- Load data ----
    print("\n--- Loading training data ---")
    X, y = load_data(min_nerve_pixels=100)
    print(f"Dataset: {X.shape[0]} samples  image {X.shape[1:]}")

    split   = int(0.9 * len(X))
    X_tr, X_vl = X[:split], X[split:]
    y_tr, y_vl = y[:split], y[split:]

    train_ds = UltrasoundDataset(X_tr, y_tr, augment=True)
    val_ds   = UltrasoundDataset(X_vl, y_vl, augment=False)

    # batch_size=8 at 256×256 is safe for 12 GB VRAM with the larger model
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True)

    # ---- Build model ----
    model  = ResUNetCBAMASPP().to(device)
    n_par  = sum(p.numel() for p in model.parameters())
    print(f"\nModel   : ResUNetCBAMASPP")
    print(f"Params  : {n_par:,}")

    # ---- Train ----
    print("\n--- Training ---")
    model = train_model(model, train_loader, val_loader, epochs=80)

    # ---- Predict ----
    print("\n--- Loading test data ---")
    X_test, names = load_test()
    print(f"Test samples: {len(X_test)}")

    print("\n--- Predicting (batched TTA) ---")
    preds = predict_batched(model, X_test, batch_size=32)

    # ---- Visualise first 3 results ----
    n_vis = min(3, len(X_test))
    for i in range(n_vis):
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(X_test[i].squeeze(), cmap="gray")
        plt.title(f"Original: {names[i]}")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(preds[i].squeeze() > 0.5, cmap="gray")
        plt.title("Predicted mask")
        plt.axis("off")
        plt.tight_layout()
        out_path = os.path.join(_SCRIPT_DIR, f"best_pred_{i}.png")
        plt.savefig(out_path)
        print(f"Saved: {out_path}")
        plt.show()

    print("\nFINISHED SUCCESSFULLY")
