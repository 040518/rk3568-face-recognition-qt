#!/bin/sh

set -eu

APP_ROOT="${APP_ROOT:-/root/rknn_face}"
SERVICE="$APP_ROOT/face-recognition-service.sh"

# SystemUI 桌面图标的唯一入口：如果服务已在运行，不再启动第二个实例。
if "$SERVICE" status >/dev/null 2>&1; then
    exit 0
fi

# 使用 exec 把图标入口进程替换为服务进程，方便 PID 文件和 stop 命令统一管理。
exec "$SERVICE" start
