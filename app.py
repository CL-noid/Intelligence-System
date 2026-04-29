"""
app.py  —  Flask 后端
.venv\Scripts\activate
运行：python app.py  →  http://localhost:5000
"""

import os
import glob
from flask import Flask, request, jsonify, send_file, send_from_directory
from io import BytesIO
import segmentation
import feature_extraction
import life_prediction

MODEL_DIR   = os.path.join(os.path.dirname(__file__), 'models')
ALLOWED_EXT = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
MAX_MB      = 200

app = Flask(__name__, static_folder='.', static_url_path='')
app.config['MAX_CONTENT_LENGTH'] = MAX_MB * 1024 * 1024

# ── 启动时加载所有模型 ─────────────────────────────
def _find_pth():
    files = glob.glob(os.path.join(MODEL_DIR, '*.pth'))
    return files[0] if files else None

_pth = _find_pth()
if _pth:
    segmentation.load_model(_pth)
else:
    print(f"[app] WARNING: models/ 中无 .pth 文件 — Module 1 已禁用。")

# UMAP 模型（Module 2 需要）
try:
    feature_extraction.load_umap_model(MODEL_DIR)
except Exception as e:
    print(f"[app] WARNING: UMAP 模型加载失败 — Module 2 UMAP 步骤已禁用。({e})")

# GPR 模型 + UMAP 归一化参数（Module 3 & 4 需要）
try:
    life_prediction.load_life_model(MODEL_DIR)
except Exception as e:
    print(f"[app] WARNING: GPR 模型加载失败 — Modules 3 & 4 已禁用。({e})")

# ── 静态文件 ──────────────────────────────────────
@app.route('/')
def index():   return send_from_directory('.', 'index.html')
@app.route('/style.css')
def css():     return send_from_directory('.', 'style.css')
@app.route('/main.js')
def js():      return send_from_directory('.', 'main.js')

# ── Module 1 — 图像分割 ───────────────────────────
@app.route('/api/status')
def status():
    return jsonify({
        'model_loaded': segmentation.is_model_loaded(),
        'model_file':   os.path.basename(_pth) if _pth else None,
    })

@app.route('/api/segment', methods=['POST'])
def segment_route():
    if not segmentation.is_model_loaded():
        return jsonify({'error': '分割模型未加载，请将 .pth 文件放入 models/ 后重启。'}), 503
    if 'image' not in request.files:
        return jsonify({'error': '请求中没有图像文件。'}), 400
    file = request.files['image']
    ext  = os.path.splitext(file.filename or '')[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify({'error': f'不支持的格式: {ext}'}), 400
    try:
        result_bytes = segmentation.run_segmentation(file.read())
        return send_file(BytesIO(result_bytes), mimetype='image/png',
                         download_name='segmentation_result.png')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── Module 2 — 特征提取 ───────────────────────────
@app.route('/api/extract', methods=['POST'])
def extract_route():
    files = request.files.getlist('images')
    if not files or files[0].filename == '':
        return jsonify({'error': '请至少上传一张图像。'}), 400
    magnification = request.form.get('magnification', '8k').strip()
    angle_str     = request.form.get('angle', '0').strip()
    if magnification not in ('3k', '5k', '8k'):
        return jsonify({'error': f'放大倍数无效: {magnification}'}), 400
    if angle_str not in ('0', '45', '90'):
        return jsonify({'error': f'角度无效: {angle_str}'}), 400
    for f in files:
        if os.path.splitext(f.filename or '')[1].lower() not in ALLOWED_EXT:
            return jsonify({'error': f'不支持的格式: {f.filename}'}), 400
    images_bytes = [f.read() for f in files]
    filenames    = [f.filename for f in files]
    try:
        results = feature_extraction.extract_features(
            images_bytes=images_bytes,
            magnification=magnification,
            angle_deg=int(angle_str),
        )
        for i, r in enumerate(results):
            r['filename'] = filenames[i]
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── Module 3 — 手动输入特征预测 ───────────────────
@app.route('/api/predict', methods=['POST'])
def predict_route():
    """
    JSON 字段：
      temp        — 温度 (°C)
      stress      — 应力 (MPa)
      umap1_norm  — UMAP1 已归一化值（来自 Module 2 输出）
      umap2_norm  — UMAP2 已归一化值
      omega_norm  — ω（微米，Module 2 输出的 omega_norm 即微米值）
      l_norm      — l（微米，同上）
    """
    if not life_prediction.is_life_model_loaded():
        return jsonify({'error': 'GPR 模型未加载。'}), 503
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': '请求体必须为 JSON 格式。'}), 400
    required = ['temp', 'stress', 'umap1_norm', 'umap2_norm', 'omega_norm', 'l_norm']
    missing  = [k for k in required if k not in data]
    if missing:
        return jsonify({'error': f'缺少字段: {", ".join(missing)}'}), 400
    try:
        result = life_prediction.predict_life(
            temp_c     = float(data['temp']),
            stress_mpa = float(data['stress']),
            umap1_norm = float(data['umap1_norm']),
            umap2_norm = float(data['umap2_norm']),
            omega_um   = float(data['omega_norm']),   # 前端字段名保持不变，后端用微米
            l_um       = float(data['l_norm']),
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── Module 4 — 图像一键预测 ──────────────────────
@app.route('/api/predict_full', methods=['POST'])
def predict_full_route():
    """
    Form 字段：
      images        — 一张或多张二值化图像
      temp          — 温度 (°C)
      stress        — 应力 (MPa)
      umap1_norm    — UMAP1 已归一化值（来自 Module 2，所有图像共用）
      umap2_norm    — UMAP2 已归一化值
      magnification — '3k'|'5k'|'8k'
      angle         — '0'|'45'|'90'
    返回 JSON 列表，每张图一个条目：
      [{ filename, life_h, life_std_h, life_low, life_high,
         omega_norm, l_norm, volume_fraction }, ...]
    """
    if not life_prediction.is_life_model_loaded():
        return jsonify({'error': 'GPR 模型未加载。'}), 503

    files = request.files.getlist('images')
    if not files or files[0].filename == '':
        return jsonify({'error': '请至少上传一张图像。'}), 400

    try:
        temp_c     = float(request.form.get('temp', ''))
        stress_mpa = float(request.form.get('stress', ''))
    except (ValueError, TypeError):
        return jsonify({'error': '温度、应力必须为数字。'}), 400

    magnification = request.form.get('magnification', '8k').strip()
    angle_str     = request.form.get('angle', '0').strip()
    if magnification not in ('3k', '5k', '8k'):
        return jsonify({'error': f'放大倍数无效: {magnification}'}), 400
    if angle_str not in ('0', '45', '90'):
        return jsonify({'error': f'角度无效: {angle_str}'}), 400

    for f in files:
        if os.path.splitext(f.filename or '')[1].lower() not in ALLOWED_EXT:
            return jsonify({'error': f'不支持的格式: {f.filename}'}), 400

    images_bytes = [f.read() for f in files]
    filenames    = [f.filename for f in files]

    try:
        from feature_extraction import _decode_gray, _run_cca, _run_cld, _run_umap_transform, L_PIXS_MAP

        L_pixs = L_PIXS_MAP.get(magnification.lower(), L_PIXS_MAP['8k'])
        angle  = int(angle_str)

        # Step 1: 解码 + CCA（获取特征向量用于 UMAP）
        decoded     = [_decode_gray(b) for b in images_bytes]
        cca_results = [_run_cca(img) for img in decoded]
        feat_vecs   = [r[0] for r in cca_results]
        cca_volumes = [r[1] for r in cca_results]

        # Step 2: UMAP transform（用训练好的模型，自动归一化）
        import numpy as np
        X = np.array(feat_vecs)
        u1_raw, u2_raw, u1_norm, u2_norm = _run_umap_transform(X)

        # Step 3: CLD + GPR 预测
        results = []
        for i, img in enumerate(decoded):
            gw, gpw, _ = _run_cld(img, L_pixs, angle)

            res = life_prediction.predict_life(
                temp_c     = temp_c,
                stress_mpa = stress_mpa,
                umap1_norm = float(u1_norm[i]),
                umap2_norm = float(u2_norm[i]),
                omega_um   = gw,
                l_um       = gpw,
            )
            res['filename']        = filenames[i]
            res['omega_norm']      = round(gw,  4)
            res['l_norm']          = round(gpw, 4)
            res['umap1_norm']      = round(float(u1_norm[i]), 4)
            res['umap2_norm']      = round(float(u2_norm[i]), 4)
            res['volume_fraction'] = round(cca_volumes[i], 4)
            results.append(res)

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*55)
    print("  涡轮叶片智能评价系统  —  后端服务")
    print("="*55)
    print(f"  访问地址:  http://localhost:5000")
    print(f"  模型目录:  {MODEL_DIR}")
    print("="*55 + "\n")
    app.run(debug=False, host='0.0.0.0', port=5000)
