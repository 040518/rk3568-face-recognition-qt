#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ADB="${ADB:-adb}"
REMOTE_DIR="/root/rknn_face"

"$ADB" get-state >/dev/null
"$ADB" push "$PROJECT_ROOT/board/rknn-face-landscape.sh" \
    /etc/profile.d/rknn-face-landscape.sh
"$ADB" push "$PROJECT_ROOT/board/S49z-face-landscape" \
    /etc/init.d/S49z-face-landscape
"$ADB" push "$PROJECT_ROOT/board/start_weston_landscape.sh" \
    "$REMOTE_DIR/start_weston_landscape.sh"
"$ADB" shell chmod 755 \
    /etc/profile.d/rknn-face-landscape.sh \
    /etc/init.d/S49z-face-landscape \
    "$REMOTE_DIR/start_weston_landscape.sh"
"$ADB" shell "$REMOTE_DIR/start_weston_landscape.sh"

printf 'Weston started in native desktop mode: 720x1280.\n'
printf 'The face-recognition launcher switches to 1280x720 only while active.\n'
