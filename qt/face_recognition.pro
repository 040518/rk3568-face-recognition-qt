QT += core gui widgets

CONFIG += c++17 console link_pkgconfig
CONFIG -= app_bundle

TEMPLATE = app
TARGET = face_recognition.aarch64

PKGCONFIG += opencv4

# Point this variable at Rockchip's public RKNPU2 librknn_api directory, for
# example: runtime/RK356X/Linux/librknn_api
RKNN_RUNTIME_DIR = $$(RKNN_RUNTIME_DIR)
isEmpty(RKNN_RUNTIME_DIR) {
    error("Set RKNN_RUNTIME_DIR to the RKNPU2 librknn_api directory")
}

INCLUDEPATH += \
    $$PWD/../include \
    $$PWD/../modules/v4l2_capture/include \
    $$RKNN_RUNTIME_DIR/include

SOURCES += \
    $$PWD/../src/main.cpp \
    $$PWD/../src/mainwindow.cpp \
    $$PWD/../src/camera_worker.cpp \
    $$PWD/../src/face_engine.cpp \
    $$PWD/../modules/v4l2_capture/src/v4l2_capture.cpp

HEADERS += \
    $$PWD/../include/mainwindow.h \
    $$PWD/../include/camera_worker.h \
    $$PWD/../include/face_engine.h \
    $$PWD/../modules/v4l2_capture/include/v4l2_capture.h

LIBS += -L$$RKNN_RUNTIME_DIR/aarch64 -lrknnrt

QMAKE_CXXFLAGS += -Wall -Wextra -Wpedantic -O2

