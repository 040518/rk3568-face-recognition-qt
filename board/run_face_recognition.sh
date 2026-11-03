#!/bin/sh

set -eu

APP_ROOT="${APP_ROOT:-/root/rknn_face}"
RUNTIME_DIR="/tmp/rknn-face-runtime"
LOG_FILE="${LOG_FILE:-$APP_ROOT/face_recognition.log}"
TRANSFORM_FILE="${WESTON_DRM_CONFIG:-/tmp/.weston_drm.conf}"
APP_PID=""
WAYLAND_ACTIVE=0

restore_desktop() {
    # 无论正常退出、Ctrl+C 或服务停止，都先结束应用，再恢复系统桌面的竖屏方向。
    if [ -n "$APP_PID" ] && kill -0 "$APP_PID" 2>/dev/null; then
        kill "$APP_PID" 2>/dev/null || true
        wait "$APP_PID" 2>/dev/null || true
    fi
    if [ "$WAYLAND_ACTIVE" -eq 1 ]; then
        printf '%s\n' 'output:all:rotate0' >"$TRANSFORM_FILE"
    fi
}

trap restore_desktop EXIT
trap 'exit 143' HUP INT TERM

if [ -S /run/wayland-0 ]; then
    # Buildroot/Weston 的 Wayland socket 不在应用私有运行目录中，创建仅含软链接的目录供 Qt 使用。
    mkdir -p "$RUNTIME_DIR"
    chmod 0700 "$RUNTIME_DIR"
    ln -sf /run/wayland-0 "$RUNTIME_DIR/wayland-0"
    export XDG_RUNTIME_DIR="$RUNTIME_DIR"
    export QT_QPA_PLATFORM="wayland"
    WAYLAND_ACTIVE=1
    # 只在识别程序运行期间临时切为横屏；退出时由 trap 写回 rotate0。
    printf '%s\n' 'output:all:rotate90' >"$TRANSFORM_FILE"
    # The vendor system UI recreates dashboard/calendar surfaces after an
    # output transform. Wait until those surfaces settle so this fullscreen
    # application is the final top-level surface.
    sleep 6
else
    # 没有 Weston 时走 framebuffer 回退路径，便于纯控制台/调试环境使用。
    export QT_QPA_PLATFORM="linuxfb:fb=/dev/fb0"
fi

# "$@" 保留服务脚本传入的 --image、--interval 等调试参数。
"$APP_ROOT/face_recognition" \
    --det-model "$APP_ROOT/models/det_500m_rk3568_int8.rknn" \
    --rec-model "$APP_ROOT/models/w600k_mbf_rk3568_int8_mmse.rknn" \
    --gallery "$APP_ROOT/gallery" \
    --camera /dev/video0 \
    --camera-rotation left \
    "$@" >>"$LOG_FILE" 2>&1 &
APP_PID=$!
wait "$APP_PID"
APP_STATUS=$?
APP_PID=""
exit "$APP_STATUS"
