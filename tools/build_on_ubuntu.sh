#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SDK_ROOT="${SDK_ROOT:?Set SDK_ROOT to the RK3568 Buildroot host toolchain directory}"
RKNN_RUNTIME_DIR="${RKNN_RUNTIME_DIR:?Set RKNN_RUNTIME_DIR to the RKNPU2 librknn_api directory}"
QMAKE="${QMAKE:-$SDK_ROOT/bin/qmake}"
BUILD_DIR="$PROJECT_ROOT/build-qt"

if [ ! -x "$QMAKE" ]; then
    printf 'ERROR: qmake not found: %s\n' "$QMAKE" >&2
    exit 1
fi
if [ ! -f "$RKNN_RUNTIME_DIR/include/rknn_api.h" ] || \
   [ ! -f "$RKNN_RUNTIME_DIR/aarch64/librknnrt.so" ]; then
    printf 'ERROR: invalid RKNN_RUNTIME_DIR: %s\n' "$RKNN_RUNTIME_DIR" >&2
    exit 1
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
export RKNN_RUNTIME_DIR
"$QMAKE" "$PROJECT_ROOT/qt/face_recognition.pro"
make -j"$(nproc)"
file "$BUILD_DIR/face_recognition.aarch64"

