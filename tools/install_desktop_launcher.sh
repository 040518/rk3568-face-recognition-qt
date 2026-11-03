#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ADB="${ADB:-adb}"
REMOTE_DIR="/root/rknn_face"
SYSTEMUI_ROOT="/opt/ui/src"
CONFIG="$SYSTEMUI_ROOT/ATK-DLRK3568/apk1.cfg"
ENTRY="appicons/face_recognition.png 人脸识别 face_recognition"
ICON="$PROJECT_ROOT/assets/face_recognition_icon.png"

if [ ! -f "$ICON" ]; then
    printf 'Missing desktop icon: %s\n' "$ICON" >&2
    exit 1
fi

"$ADB" get-state >/dev/null
"$ADB" push "$PROJECT_ROOT/board/launch_face_recognition.sh" \
    "$REMOTE_DIR/launch_face_recognition.sh"
"$ADB" shell "chmod 755 '$REMOTE_DIR/launch_face_recognition.sh'"
"$ADB" push "$ICON" \
    "$SYSTEMUI_ROOT/appicons/face_recognition.png"
"$ADB" shell "ln -sf '$REMOTE_DIR/launch_face_recognition.sh' \
    '$SYSTEMUI_ROOT/apps/face_recognition'"

"$ADB" shell "grep -Fq 'face_recognition' '$CONFIG' || \
    printf '%s\\n' '$ENTRY' >>'$CONFIG'"

"$ADB" shell ". /etc/profile; \
    killall systemui 2>/dev/null || true; \
    killall sysvolume 2>/dev/null || true; \
    start-stop-daemon -S -b -x '$SYSTEMUI_ROOT/apps/sysvolume'; \
    start-stop-daemon -S -b -x /opt/ui/systemui; sleep 2"

printf 'Installed desktop launcher: 人脸识别\n'
