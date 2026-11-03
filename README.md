# RK3568 Face Recognition Qt

在 RK3568 上运行的本地人脸识别应用：MIPI 摄像头通过 V4L2 输出 NV12，
SCRFD 完成人脸检测和五点定位，MobileFaceNet/ArcFace 提取 512 维特征，
最后由 Qt 5 在 5.5 英寸 MIPI 屏上显示姓名、相似度、FPS 和阶段耗时。

```text
MIPI Camera -> V4L2/NV12 -> Rotate left 90 degrees
            -> SCRFD INT8 (RKNN) -> 5-point alignment
            -> MobileFaceNet MMSE INT8 (RKNN) -> cosine matching
            -> Qt/Wayland, 1280x720 landscape MIPI UI
```

## 功能

- C++17 正式运行时，Python 仅用于模型转换、量化和离线验证。
- V4L2 采集、RKNN 推理和 Qt GUI 分线程运行，只保留最新一帧，避免延迟累积。
- `gallery/<姓名>/*.jpg` 本地人脸库，多张照片特征归一化求均值。
- 开始/停止、右上角退出、桌面图标启动，以及异常退出后恢复系统桌面方向。
- 静态图片验收模式和摄像头隐私保护：实时模式拒绝保存屏幕截图。
- 检测 INT8 与识别 MMSE INT8 的量化回归工具和验证数据。

## 目录

- `src/`、`include/`：程序入口、Qt 界面、工作线程和人脸推理引擎。
- `modules/v4l2_capture/`：V4L2/NV12 摄像头采集模块。
- `qt/face_recognition.pro`：Qt 5/qmake 工程。
- `tools/`：模型转换、校准、回归、交叉编译和 ADB 部署脚本。
- `board/`：板端服务、横竖屏切换和桌面启动入口。
- `models/`：模型下载与校验说明，不包含第三方模型二进制。
- `gallery/`：人脸库格式说明，不包含任何真实人脸。
- `docs/quantization-report.md`：FP16/INT8 回归结果。
- `assets/face_recognition_icon.png`：板端桌面图标。

本公开仓库不包含实验报告、学生资料、登记照、embedding、运行日志、旧项目
源码、板厂保密头文件/运行库或第三方模型权重。

## 硬件与软件环境

本项目已在以下组合上验证：

- RK3568，Rockchip RKNPU 驱动 0.9.2。
- RKNPU2 Runtime 2.0.0b0。
- Qt 5 Widgets、Wayland/Weston、OpenCV 4。
- `/dev/video0`，NV12 1280x720 MIPI 摄像头。
- 物理 720x1280 MIPI 屏，应用运行期切换为逻辑 1280x720。

不同板卡镜像的驱动、RKNPU2 Runtime 和交叉编译工具链必须彼此匹配。

## 获取模型

下载 InsightFace `buffalo_sc`，提取以下文件：

- `det_500m.onnx`
- `w600k_mbf.onnx`

模型地址：

https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip

将 ONNX 文件放到 `models/original/`，准备代表性校准图像后运行：

```bash
python3 tools/prepare_calibration.py --help
python3 tools/convert_models.py --help
python3 tools/tune_recognizer.py --help
python3 tools/validate_rknn.py --help
```

默认部署模型为：

- `models/rknn/det_500m_rk3568_int8.rknn`
- `models/rknn/w600k_mbf_rk3568_int8_mmse.rknn`

InsightFace 代码采用 MIT 许可证，但其官方预训练权重只允许非商业研究使用。
商业使用前必须另行取得权重授权或换用许可兼容的模型。详见
`THIRD_PARTY_NOTICES.md`。

## 交叉编译

先从 Rockchip RKNPU2 仓库或板卡 SDK 获取与板端匹配的 `librknn_api` 目录，
其中应包含：

```text
include/rknn_api.h
aarch64/librknnrt.so
```

在 Ubuntu SDK 主机设置环境变量后编译：

```bash
git clone https://github.com/040518/rk3568-face-recognition-qt.git
cd rk3568-face-recognition-qt

export SDK_ROOT=/path/to/buildroot/output/host
export RKNN_RUNTIME_DIR=/path/to/rknpu2/runtime/RK356X/Linux/librknn_api
bash tools/build_on_ubuntu.sh
```

编译产物位于 `build-qt/face_recognition.aarch64`。

## 部署到 RK3568

确保 `adb devices` 能看到板卡，且 `adb` 已加入 `PATH`。然后执行：

```bash
bash tools/deploy_to_board.sh
bash tools/install_landscape.sh
bash tools/install_desktop_launcher.sh
```

板端目录默认为 `/root/rknn_face`。可直接点击系统桌面的“人脸识别”图标，
也可以通过服务脚本控制：

```bash
/root/rknn_face/face-recognition-service.sh start
/root/rknn_face/face-recognition-service.sh status
/root/rknn_face/face-recognition-service.sh stop
```

程序运行时将桌面输出临时切换为横屏；点击右上角“×”、停止服务或程序异常
退出后会恢复原生竖屏。实时摄像头默认在推理前向左旋转 90°，也可用
`--camera-rotation none|left|right|180` 覆盖。

## 添加人脸

只登记已获得明确授权的人员。每个人建议使用 3–5 张正面、轻微左右转头、
不同光照的清晰照片：

```text
gallery/
  person_a/
    01.jpg
    02.jpg
  person_b/
    01.jpg
```

部署脚本会把 `gallery/` 同步到板端。公开提交前请确认照片目录仍被
`.gitignore` 排除。默认余弦阈值 `0.45` 只是起点，应使用自己的授权验证集标定。

## 静态界面验收

摄像头被遮挡或需要先检查排版时，可自行准备一张有使用权的测试图并推送：

```bash
adb shell mkdir -p /root/rknn_face/assets
adb push /path/to/test.jpg /root/rknn_face/assets/test.jpg
/root/rknn_face/face-recognition-service.sh restart \
  --image /root/rknn_face/assets/test.jpg
```

仅在 `--image` 模式下可保存 Qt 验收图：

```bash
/root/rknn_face/face-recognition-service.sh restart \
  --image /root/rknn_face/assets/test.jpg \
  --screenshot /root/rknn_face/validation-ui.png
```

摄像头模式会拒绝 `--screenshot`，防止意外保存实时人脸画面。

## 性能与摄像头帧率

当前 BSP 的 IMX415 驱动只公开 20/30 FPS 模式，实机通常约 25 FPS；强制
请求 60 FPS 不会改变驱动模式。因此项目没有伪设 60 FPS，默认使用
`--interval 2` 让推理约为 12.5 FPS。要达到 60 FPS，必须由模组厂商提供与
当前 MIPI Lane、时钟、RAW 位深和 RKAIQ 配套的驱动模式表。

量化数据见 `docs/quantization-report.md`。普通 INT8 识别模型未达到严格回归
门槛，MMSE INT8 最低余弦达到 `0.990395` 后才被设为默认模型。

## 隐私与安全边界

- 人脸照片和 embedding 只保存在本地设备，不上传网络。
- 本项目不包含活体检测，不适合作为高安全等级门禁系统。
- 请遵守当地个人信息和生物识别数据法规。

## 许可证

本项目原创代码采用 MIT License。第三方模型、运行库和依赖保留各自许可证；
详情见 `THIRD_PARTY_NOTICES.md`。
