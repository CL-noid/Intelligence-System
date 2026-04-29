# 涡轮叶片智能评价系统 — 启动说明

## 文件结构
```
project/
├── app.py            ← Flask 后端（启动入口）
├── segmentation.py   ← 核心 ML 推理逻辑
├── index.html        ← 前端主页面
├── style.css         ← 样式
├── main.js           ← 前端交互逻辑
├── models/           ← 【把你的 .pth 文件放在这里】
│   └── your_model.pth
└── README.md
```

## 第一步：安装依赖（只需执行一次）
```bash
pip install flask torch torchvision opencv-python pillow numpy
```

## 第二步：放入模型文件
将训练好的 `.pth` 文件（如 `unet_depth4_ch32_epochs200.pth`）复制到 `models/` 文件夹内。

## 第三步：启动服务
在项目根目录打开终端，执行：
```bash
python app.py
```
终端会显示：
```
访问地址:  http://localhost:5000
```

## 第四步：打开浏览器
访问 http://localhost:5000，点击"图像阈值分割"模块即可使用。

## 使用方法（Module 1）
1. 点击或拖拽图像到上传区域（支持 PNG / JPG / BMP / TIF）
2. 选择输出模式（彩色可视化 或 二值掩码）
3. 点击"运行分割"
4. 等待推理完成后查看结果，点击"下载结果"保存 PNG

## 说明
- 模型使用 UNet4Layer（4层，base_channels=32），与训练代码完全一致
- 推理流程：将输入图像切成 512×512 分块 → 逐块推理 → 按原始顺序拼合
- 如图像尺寸不是 512 的倍数，边缘余量部分不参与推理（与 新分割.py 行为一致）
