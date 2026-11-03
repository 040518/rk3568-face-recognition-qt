#!/bin/sh

set -eu

APP_ROOT="${APP_ROOT:-/root/rknn_face}"
PID_FILE="/var/run/rknn-face.pid"

is_running() {
    # PID 文件存在并不代表进程仍活着，必须再用 kill -0 做无副作用检查。
    [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

case "${1:-status}" in
    start)
        shift
        if is_running; then
            printf 'already running: pid=%s\n' "$(cat "$PID_FILE")"
            exit 0
        fi
        rm -f "$PID_FILE"
        # -b 后台运行，-m 维护 PID 文件；run 脚本负责应用退出时恢复桌面方向。
        start-stop-daemon -S -b -m -p "$PID_FILE" \
            -x "$APP_ROOT/run_face_recognition.sh" -- "$@"
        sleep 0.5
        if ! is_running; then
            rm -f "$PID_FILE"
            printf 'start failed\n' >&2
            exit 1
        fi
        printf 'started: pid=%s\n' "$(cat "$PID_FILE")"
        ;;
    stop)
        if ! is_running; then
            rm -f "$PID_FILE"
            printf 'not running\n'
            exit 0
        fi
        app_pid="$(cat "$PID_FILE")"
        kill "$app_pid"
        wait_count=0
        # 先给程序 3 秒正常退出机会，使 Qt、V4L2 与 RKNN 走清理逻辑；超时才强制结束。
        while kill -0 "$app_pid" 2>/dev/null && [ "$wait_count" -lt 30 ]; do
            sleep 0.1
            wait_count=$((wait_count + 1))
        done
        if kill -0 "$app_pid" 2>/dev/null; then
            kill -KILL "$app_pid"
        fi
        rm -f "$PID_FILE"
        printf 'stopped\n'
        ;;
    restart)
        shift
        "$0" stop
        "$0" start "$@"
        ;;
    status)
        if is_running; then
            printf 'running: pid=%s\n' "$(cat "$PID_FILE")"
        else
            printf 'stopped\n'
            exit 1
        fi
        ;;
    *)
        printf 'usage: %s {start|stop|restart|status}\n' "$0" >&2
        exit 2
        ;;
esac
