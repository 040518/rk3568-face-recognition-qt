#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ADB="${ADB:-adb}"
REMOTE_DIR="/root/rknn_face"
BINARY="$PROJECT_ROOT/build-qt/face_recognition.aarch64"

if [ ! -x "$ADB" ]; then
    printf 'ERROR: adb not found: %s\n' "$ADB" >&2
    exit 1
fi
if [ ! -f "$BINARY" ]; then
    printf 'ERROR: build first: %s\n' "$BINARY" >&2
    exit 1
fi

# 先确认 ADB 连接正常，再开始覆盖板端程序、脚本和模型。
"$ADB" get-state >/dev/null
"$ADB" shell "mkdir -p '$REMOTE_DIR/models' '$REMOTE_DIR/gallery'"
# 可执行文件在板端使用固定名称；run_face_recognition.sh 默认从该位置启动它。
"$ADB" push "$BINARY" "$REMOTE_DIR/face_recognition"
"$ADB" push "$PROJECT_ROOT/board/run_face_recognition.sh" \
    "$REMOTE_DIR/run_face_recognition.sh"
"$ADB" push "$PROJECT_ROOT/board/face-recognition-service.sh" \
    "$REMOTE_DIR/face-recognition-service.sh"
"$ADB" push "$PROJECT_ROOT/board/launch_face_recognition.sh" \
    "$REMOTE_DIR/launch_face_recognition.sh"
"$ADB" shell "chmod 755 '$REMOTE_DIR/face_recognition'"
"$ADB" shell "chmod 755 '$REMOTE_DIR/run_face_recognition.sh'"
"$ADB" shell "chmod 755 '$REMOTE_DIR/face-recognition-service.sh'"
"$ADB" shell "chmod 755 '$REMOTE_DIR/launch_face_recognition.sh'"

# 两个模型必须成对部署：检测为 INT8，识别为已通过回归的 MMSE INT8。
for model in \
    det_500m_rk3568_int8.rknn \
    w600k_mbf_rk3568_int8_mmse.rknn; do
    local_model="$PROJECT_ROOT/models/rknn/$model"
    if [ ! -f "$local_model" ]; then
        printf 'ERROR: missing model: %s\n' "$local_model" >&2
        exit 1
    fi
    "$ADB" push "$local_model" "$REMOTE_DIR/models/$model"
done

# gallery/<姓名>/ 下仅放置已授权登记照；部署脚本不从网络下载或上传人脸数据。
if [ -d "$PROJECT_ROOT/gallery" ]; then
    "$ADB" push "$PROJECT_ROOT/gallery/." "$REMOTE_DIR/gallery/"
fi

printf 'Deployed to %s\n' "$REMOTE_DIR"
printf 'Run on board:\n'
printf '  %s/run_face_recognition.sh\n' "$REMOTE_DIR"
printf 'Background service control:\n'
printf '  %s/face-recognition-service.sh {start|stop|restart|status}\n' \
    "$REMOTE_DIR"

