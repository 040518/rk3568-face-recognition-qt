#!/bin/sh

set -e

. /etc/profile

killall weston 2>/dev/null || true
rm -f /run/wayland-0 /run/wayland-0.lock
printf '%s\n' 'output:all:rotate0' >"$WESTON_DRM_CONFIG"
: > /root/rknn_face/weston.log
nohup /usr/bin/weston >>/root/rknn_face/weston.log 2>&1 </dev/null &
weston_pid=$!
sleep 2
kill -0 "$weston_pid"
printf 'weston_started pid=%s desktop=720x1280\n' "$weston_pid"
