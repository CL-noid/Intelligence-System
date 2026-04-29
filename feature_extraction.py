"""
feature_extraction.py
Module 2: 图像特征提取
Pipeline: CCA → UMAP（调用训练好的 umap_model.pkl）→ min-max 归一化（umap_norm_params.pkl）
          + CLD（结果单位微米，不额外归一化）

关键变更（与旧版区别）：
  - UMAP 不再在批内 fit，改为 transform 到训练空间
  - UMAP 归一化用训练保存的 min/max，不用批内统计
  - ω、l 以微米直接输出，SIZE_NORM 归一化已移除
  - 对外接口字段名保持不变（omega_raw/norm, l_raw/norm, umap1_raw/norm, umap2_raw/norm）
    其中 omega_norm = omega_raw（微米），l_norm = l_raw（微米），
    命名保留 _norm 后缀是为了不破坏前端已有调用
"""

import numpy as np
import cv2
import os
from io import BytesIO
from numpy.fft import fft2, ifft2, fftshift
from sklearn.mixture import GaussianMixture
from typing import List, Dict, Tuple

# ─────────────────────────────────────────────
#  全局参数
# ─────────────────────────────────────────────
TARGET_SHAPE = (128, 128)   # CCA 特征图缩放尺寸
CLD_STEP     = 5            # CLD 旋转步长（度）

L_PIXS_MAP = {
    '3k': 1.0 / 87,
    '5k': 1.0 / 141,
    '8k': 1.0 / 222,
}

# ─────────────────────────────────────────────
#  UMAP 模型（由 app.py 启动时调用 load_umap_model 加载）
# ─────────────────────────────────────────────
_umap_model   = None   # 训练好的 UMAP transform 对象
_umap_norm    = None   # {'min1', 'max1', 'min2', 'max2'}


def load_umap_model(models_dir: str):
    """从 models_dir 加载 umap_model.pkl 和 umap_norm_params.pkl。"""
    global _umap_model, _umap_norm
    import joblib
    model_path  = os.path.join(models_dir, 'umap_model.pkl')
    params_path = os.path.join(models_dir, 'umap_norm_params.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到 UMAP 模型: {model_path}")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"找不到 UMAP 归一化参数: {params_path}")
    _umap_model = joblib.load(model_path)
    _umap_norm  = joblib.load(params_path)
    print(f"[feature_extraction] UMAP 模型已加载: {model_path}")
    print(f"[feature_extraction] UMAP 归一化范围: "
          f"UMAP1=[{_umap_norm['min1']:.4f}, {_umap_norm['max1']:.4f}]  "
          f"UMAP2=[{_umap_norm['min2']:.4f}, {_umap_norm['max2']:.4f}]")


def is_umap_loaded() -> bool:
    return _umap_model is not None and _umap_norm is not None


# ─────────────────────────────────────────────
#  内部工具函数
# ─────────────────────────────────────────────
def _norm_01(a: np.ndarray) -> np.ndarray:
    a = a - np.min(a)
    return a / np.max(a) if np.max(a) > 0 else a


def _decode_gray(image_bytes: bytes) -> np.ndarray:
    """将图像字节解码为 float32 灰度数组（值域 0~1）。"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        from PIL import Image
        pil = Image.open(BytesIO(image_bytes)).convert('L')
        img = np.array(pil)
    if   img.dtype == np.uint16: img = img.astype(np.float32) / 65535.0
    elif img.dtype == np.uint8:  img = img.astype(np.float32) / 255.0
    else:                        img = img.astype(np.float32) / np.max(img)
    return img


# ─────────────────────────────────────────────
#  CCA
# ─────────────────────────────────────────────
def _run_cca(img_float: np.ndarray,
             target_phase_is_white: bool = False) -> Tuple[np.ndarray, float]:
    """返回 (feature_vector [16384,], volume_fraction)。"""
    bin_orig   = (img_float > 0.5).astype(np.float32)
    bin_flip   = 1.0 - bin_orig
    S          = bin_orig.size
    F1         = fft2(bin_orig - np.mean(bin_orig))
    F2         = fft2(bin_flip - np.mean(bin_flip))
    cross_corr = fftshift(ifft2(F1 * np.conj(F2)).real) / S
    resized    = cv2.resize(_norm_01(cross_corr),
                            (TARGET_SHAPE[1], TARGET_SHAPE[0]),
                            interpolation=cv2.INTER_LINEAR)
    white_frac      = float(np.mean(bin_orig))
    volume_fraction = white_frac if target_phase_is_white else 1.0 - white_frac
    return resized.flatten(), volume_fraction


# ─────────────────────────────────────────────
#  CLD
# ─────────────────────────────────────────────
def _rotate_img(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    nw = int(h * abs(M[0,1]) + w * abs(M[0,0]))
    nh = int(h * abs(M[0,0]) + w * abs(M[0,1]))
    M[0,2] += (nw-w)/2;  M[1,2] += (nh-h)/2
    return cv2.warpAffine(img, M, (nw, nh))


def _cld_at_angle(bin_image: np.ndarray, angle: float,
                  L_pixs: float, m: int, n: int) -> Tuple[float, float]:
    """单角度 CLD，返回 (gamma_width μm, gamma_p_width μm)。"""
    hm, hn = m//2, n//2
    fill = np.zeros((m*2, n*2))
    fill[hm:hm+m, hn:hn+n]   = bin_image
    fill[0:hm,    hn:hn+n]   = bin_image[0:hm,  0:n]
    fill[hm+m:2*m,hn:hn+n]   = bin_image[hm:m,  0:n]
    fill[hm:hm+m, 0:hn]      = bin_image[0:m,   0:hn]
    fill[hm:hm+m, hn+n:2*n]  = bin_image[0:m,   hn:n]

    rot = _rotate_img(fill, angle)
    rm, rn = rot.shape
    cut = np.trunc(rot[rm//2-hm:rm//2+hm, rn//2-hn:rn//2+hn])

    front = np.zeros((150, n//10+1))
    back  = np.zeros((150, n//10+1))
    q = 0
    for j in range(0, n, 10):
        k = kk = 0
        for i in range(1, m-1):
            if cut[i-1,j]==1 and cut[i,j]==0 and cut[i+1,j]==0: front[k,q]=i;  k+=1
            if cut[i-1,j]==0 and cut[i,j]==0 and cut[i+1,j]==1: back[kk,q]=i; kk+=1
        q += 1

    K, Q = front.shape
    gamma   = np.zeros((K, Q))
    gamma_p = np.zeros((K, Q))
    for i in range(Q):
        for j in range(K-1):
            if back[0,i] > front[0,i]:
                if back[j,i]>0:    gamma_p[j,i] = L_pixs*(back[j,i]  - front[j,i])
                if front[j+1,i]>0: gamma[j,i]   = L_pixs*(front[j+1,i]-back[j,i])
            else:
                if back[j+1,i]>0:  gamma_p[j,i] = L_pixs*(back[j+1,i]-front[j,i])
                if front[j,i]>0:   gamma[j,i]   = L_pixs*(front[j,i]  -back[j,i])

    def gmm_mean(arr, lo, hi):
        arr = arr.ravel(); arr = arr[(arr>0)&(arr>=lo)&(arr<=hi)]
        if len(arr) == 0: return float('nan')
        mdl = GaussianMixture(n_components=1, covariance_type='full', random_state=0)
        mdl.fit(arr.reshape(-1,1))
        return float(np.mean(mdl.means_))

    return gmm_mean(gamma, 0.003, 1.0), gmm_mean(gamma_p, 0.125, 2.0)


def _run_cld(img_float: np.ndarray, L_pixs: float,
             selected_angle: int) -> Tuple[float, float, float]:
    """全角度 CLD 扫描，返回 (gamma_width μm, gamma_p_width μm, volume_fraction)。"""
    bin_img = (img_float > 0.5).astype(np.float32)
    m, n    = bin_img.shape
    vol     = float(1.0 - bin_img.sum() / (m*n))

    step_num = int(360 / CLD_STEP)
    out_g  = np.zeros(step_num)
    out_gp = np.zeros(step_num)

    for idx, theta in enumerate(range(0, 180, CLD_STEP)):
        gw, gpw = _cld_at_angle(bin_img, theta, L_pixs, m, n)
        out_g[idx]              = gw  if not np.isnan(gw)  else 0
        out_g[idx+step_num//2]  = gw  if not np.isnan(gw)  else 0
        out_gp[idx]             = gpw if not np.isnan(gpw) else 0
        out_gp[idx+step_num//2] = gpw if not np.isnan(gpw) else 0

    ai = int(selected_angle / CLD_STEP) % step_num
    return float(out_g[ai]), float(out_gp[ai]), vol


# ─────────────────────────────────────────────
#  UMAP（使用训练好的模型）
# ─────────────────────────────────────────────
def _run_umap_transform(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                        np.ndarray, np.ndarray]:
    """
    用训练好的 umap_model.pkl transform CCA 特征，
    再用 umap_norm_params.pkl 做 min-max 归一化。
    features: shape (N, 16384)
    返回: u1_raw, u2_raw, u1_norm, u2_norm  各 shape (N,)
    """
    if not is_umap_loaded():
        raise RuntimeError("UMAP 模型未加载，请检查 models/ 目录。")

    emb    = _umap_model.transform(features)        # shape (N, 2)
    u1_raw = emb[:, 0]
    u2_raw = emb[:, 1]

    u1_norm = np.clip(
        (u1_raw - _umap_norm['min1']) / (_umap_norm['max1'] - _umap_norm['min1']),
        0.0, 1.0)
    u2_norm = np.clip(
        (u2_raw - _umap_norm['min2']) / (_umap_norm['max2'] - _umap_norm['min2']),
        0.0, 1.0)
    return u1_raw, u2_raw, u1_norm, u2_norm


# ─────────────────────────────────────────────
#  公开 API（由 app.py 调用）
# ─────────────────────────────────────────────
def extract_features(
    images_bytes: List[bytes],
    magnification: str,          # '3k' | '5k' | '8k'
    angle_deg: int,              # 0 | 45 | 90
    target_phase_is_white: bool = False,
) -> List[Dict]:
    """
    处理一批图像，返回每张图的特征字典：
      omega_raw / omega_norm  — ω（微米），两者相同（不额外归一化）
      l_raw     / l_norm      — l（微米），两者相同
      umap1_raw / umap1_norm  — UMAP1 原始值 / min-max 归一化值
      umap2_raw / umap2_norm  — UMAP2 原始值 / min-max 归一化值
      volume_fraction         — 体积分数
    """
    L_pixs = L_PIXS_MAP.get(magnification.lower(), L_PIXS_MAP['8k'])

    # Step 1: 解码 + CCA
    decoded      = [_decode_gray(b) for b in images_bytes]
    cca_results  = [_run_cca(img, target_phase_is_white) for img in decoded]
    feat_vecs    = [r[0] for r in cca_results]
    cca_volumes  = [r[1] for r in cca_results]

    # Step 2: UMAP transform（用训练好的模型）
    X = np.array(feat_vecs)
    u1_raw, u2_raw, u1_norm, u2_norm = _run_umap_transform(X)

    # Step 3: CLD
    results = []
    for i, img in enumerate(decoded):
        gw, gpw, _ = _run_cld(img, L_pixs, angle_deg)

        # ω 和 l 以微米直接输出；_norm 字段保持与前端接口一致，值等于 _raw
        results.append({
            'omega_raw':       round(gw,  4),
            'omega_norm':      round(gw,  4),   # 微米，不额外归一化
            'l_raw':           round(gpw, 4),
            'l_norm':          round(gpw, 4),   # 微米，不额外归一化
            'umap1_raw':       round(float(u1_raw[i]),  4),
            'umap1_norm':      round(float(u1_norm[i]), 4),
            'umap2_raw':       round(float(u2_raw[i]),  4),
            'umap2_norm':      round(float(u2_norm[i]), 4),
            'volume_fraction': round(cca_volumes[i], 4),
        })

    return results
