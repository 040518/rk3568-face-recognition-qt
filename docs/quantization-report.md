# RK3568 模型量化回归报告

日期：2026-10-26

## 环境

- 芯片：RK3568，RKNPU 驱动 0.9.2。
- 板端 `librknnrt`：2.0.0b0。
- 转换工具：RKNN-Toolkit2 2.0.0b0+9bab5682。
- 检测模型：InsightFace `det_500m.onnx`，输入 640×640，9 个输出。
- 识别模型：InsightFace `w600k_mbf.onnx`，输入 112×112，512 维输出。

## 校准数据

- 检测：48 张由公开测试图做尺度、亮度和裁剪增强得到的图像。
- 识别：128 张经过 SCRFD 五点对齐后再做轻量增强的人脸图。

这批数据足以做工程链路和回归检查，但不能代表最终安装现场。正式部署前还需使用已授权、覆盖实际摄像头距离和光照的数据重新校准阈值。

## 回归结果

| 模型 | 指标 | 结果 | 判定 |
|---|---|---:|---|
| SCRFD INT8 | 12/12 图像人脸数量与 FP16 一致 | 12/12 | 通过 |
| SCRFD INT8 | 匹配框 IoU 中位数 | 0.9833275 | 通过（门槛 0.90） |
| SCRFD INT8 | 五点平均归一化误差 | 0.0030704 | 通过（门槛 0.015） |
| MobileFaceNet INT8 | 64 个样本的最低 FP16/INT8 embedding 余弦 | 0.973536 | 未通过（门槛 0.98） |
| MobileFaceNet INT8 | embedding 余弦中位数 | 0.986453 | 参考 |
| MobileFaceNet INT8 | 2016 对在阈值 0.45 的判定翻转 | 0 | 通过 |
| MobileFaceNet MMSE INT8 | 64 个样本的最低 FP16/INT8 embedding 余弦 | 0.990395 | 通过（门槛 0.98） |
| MobileFaceNet MMSE INT8 | embedding 余弦中位数 | 0.994958 | 通过 |
| MobileFaceNet MMSE INT8 | 2016 对在阈值 0.45 的判定翻转 | 0 | 通过 |

## 部署决定

默认组合固定为：

- 人脸检测：INT8 RKNN。
- 人脸识别：MMSE INT8 RKNN。

普通 INT8 虽然没有造成当前小验证集的阈值判定翻转，但最低同输入 embedding 余弦没有达到项目预先设定的 0.98，因此不作为默认模型。MMSE 优化把最低余弦提升到 0.990395，并在板端通过了完全相同的 64 图/2016 对回归，因此可替换 FP16；FP16 仍保留为诊断回退模型。

## 板端实时结果

20 秒真实摄像头联调中：

- V4L2：1280×720 NV12，4 个 MMAP buffer。
- 视频帧率：约 24–26 FPS。
- SCRFD INT8：约 28–35 ms/帧。
- 回退模型联调时画面有人脸，MobileFaceNet FP16：约 20–23 ms/脸。
- 最终双 INT8 启动脚本持续约 25 FPS；该 20 秒窗口内未出现人脸，因此 MMSE INT8 的相机态单脸耗时不在此处臆测，功能与精度由上述 64 图板端回归证明。
- GUI：Qt 5 Widgets 经 Weston/Wayland 显示到物理 720×1280 MIPI 面板；Rockchip Weston 动态 `rotate90` 后，Qt 逻辑屏幕确认是 1280×720。
- 摄像头帧率：IMX415 驱动 mode 表最高 30 FPS，mainpath 拒绝 `VIDIOC_S_PARM 60`；隔帧推理后实测视频约 25 FPS、推理约 12.5 FPS。
