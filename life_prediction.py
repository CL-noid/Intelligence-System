"""
life_prediction.py
Module 3: 剩余寿命预测（手动输入特征）
Module 4: 一键预测（图像→寿命，内部走完整流程）

归一化约定（与训练完全一致，勿随意修改）
─────────────────────────────────────────────
特征            处理方式
─────────────────────────────────────────────
温度 (°C)       temp   / 1500
应力 (MPa)      stress / 300
UMAP1           (raw - min1) / (max1 - min1)  ← 来自 umap_norm_params.pkl
UMAP2           (raw - min2) / (max2 - min2)  ← 来自 umap_norm_params.pkl
ω  (μm)         原始微米值，不额外归一化
l  (μm)         原始微米值，不额外归一化
─────────────────────────────────────────────
目标值          训练在 log(RUL_norm) 空间，RUL_norm = RUL_h / NORM_DENOM
反变换          RUL_h = exp(log_pred) * NORM_DENOM
─────────────────────────────────────────────
GPR 输入列顺序（共6列）:
  [umap1_norm, umap2_norm, temp_norm, stress_norm, omega_um, l_um]
"""

import os
import numpy as np

_model       = None   # GaussianProcessRegressor
_norm_params = None   # {'min1', 'max1', 'min2', 'max2'}

# ── 归一化常数 ──────────────────────────────────
TEMP_NORM   = 1500.0    # 温度归一化分母（°C）
STRESS_NORM = 300.0     # 应力归一化分母（MPa）
NORM_DENOM  = 12905.95  # RUL 反归一化分母（小时）
CI_Z        = 1.645     # 90% 置信区间系数


def load_life_model(models_dir: str):
    """
    从 models_dir 加载：
      model_GPR_UMAP1.pkl   — 训练好的 GPR 模型
      umap_norm_params.pkl  — UMAP min-max 归一化参数
    不再需要 scaler_X.joblib（已废弃）。
    """
    global _model, _norm_params
    import joblib

    model_path  = os.path.join(models_dir, 'model_GPR_UMAP1.pkl')
    params_path = os.path.join(models_dir, 'umap_norm_params.pkl')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到 GPR 模型文件: {model_path}")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"找不到 UMAP 归一化参数文件: {params_path}")

    _model       = joblib.load(model_path)
    _norm_params = joblib.load(params_path)

    print(f"[life_prediction] GPR 模型已加载: {model_path}")
    print(f"[life_prediction] UMAP 归一化参数: "
          f"UMAP1=[{_norm_params['min1']:.4f}, {_norm_params['max1']:.4f}]  "
          f"UMAP2=[{_norm_params['min2']:.4f}, {_norm_params['max2']:.4f}]")


def is_life_model_loaded() -> bool:
    return _model is not None and _norm_params is not None


def _normalize_umap(umap1_raw: float, umap2_raw: float):
    """将 UMAP 原始坐标归一化到 [0,1]，越界时 clip。"""
    u1 = float(np.clip(
        (umap1_raw - _norm_params['min1']) / (_norm_params['max1'] - _norm_params['min1']),
        0.0, 1.0))
    u2 = float(np.clip(
        (umap2_raw - _norm_params['min2']) / (_norm_params['max2'] - _norm_params['min2']),
        0.0, 1.0))
    return u1, u2


def predict_life(
    temp_c:     float,
    stress_mpa: float,
    umap1_norm: float,   # 已归一化的 UMAP1（来自 feature_extraction 或手动输入）
    umap2_norm: float,   # 已归一化的 UMAP2
    omega_um:   float,   # ω，单位微米，不额外归一化
    l_um:       float,   # l，单位微米，不额外归一化
) -> dict:
    """
    Module 3：手动输入六个特征 → 返回预测寿命（小时）
    GPR 输入列顺序: [umap1_norm, umap2_norm, temp_norm, stress_norm, omega_um, l_um]
    """
    if not is_life_model_loaded():
        raise RuntimeError("GPR 模型未加载。")

    temp_norm   = temp_c     / TEMP_NORM
    stress_norm = stress_mpa / STRESS_NORM

    X = np.array([[umap1_norm, umap2_norm, temp_norm, stress_norm, omega_um, l_um]])

    log_pred, log_std = _model.predict(X, return_std=True)

    # 反变换：log 空间 → 归一化 RUL → 小时（Delta 法估计标准差）
    rul_norm  = float(np.exp(log_pred[0]))
    life_h    = rul_norm * NORM_DENOM
    life_std  = rul_norm * float(log_std[0]) * NORM_DENOM   # Delta method

    return {
        'life_h':     round(life_h, 2),
        'life_std_h': round(life_std, 2),
        'life_low':   round(max(0.0, life_h - CI_Z * life_std), 2),   # 90% CI 下界
        'life_high':  round(life_h + CI_Z * life_std, 2),              # 90% CI 上界
    }


def predict_life_from_raw_umap(
    umap1_raw:  float,
    umap2_raw:  float,
    temp_c:     float,
    stress_mpa: float,
    omega_um:   float,
    l_um:       float,
) -> dict:
    """
    Module 4 辅助函数：接收 UMAP 原始坐标（未归一化），
    内部做 min-max 归一化后再预测。
    """
    u1_norm, u2_norm = _normalize_umap(umap1_raw, umap2_raw)
    result = predict_life(
        temp_c=temp_c, stress_mpa=stress_mpa,
        umap1_norm=u1_norm, umap2_norm=u2_norm,
        omega_um=omega_um, l_um=l_um,
    )
    result['umap1_norm'] = round(u1_norm, 4)
    result['umap2_norm'] = round(u2_norm, 4)
    return result
