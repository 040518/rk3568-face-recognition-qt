#pragma once

#include <QMainWindow>

#include "camera_worker.h"

class QLabel;
class QPushButton;
class QTimer;
class QCloseEvent;

// Qt 主窗口只负责界面交互和显示，绝不在主线程进行摄像头读取或 RKNN 推理。
class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(const CameraWorkerConfig& config,
                        QWidget* parent = nullptr);
    ~MainWindow() override;

protected:
    // 用户点击右上角“×”时先停止工作线程，再允许窗口关闭。
    void closeEvent(QCloseEvent* event) override;

private slots:
    void start_pipeline();
    void stop_pipeline();
    // 约每 33 ms 拉取一次后台的最新帧和统计信息，不等待推理完成。
    void refresh_ui();
    void show_error(const QString& message);

private:
    QLabel* make_card(const QString& text,
                      const QString& border_color);
    void set_running_state(bool running);

    CameraWorkerConfig config_;
    CameraWorker* worker_ = nullptr;
    QTimer* refresh_timer_ = nullptr;
    QLabel* camera_label_ = nullptr;
    QLabel* name_label_ = nullptr;
    QLabel* similarity_label_ = nullptr;
    QLabel* fps_label_ = nullptr;
    QLabel* timing_label_ = nullptr;
    QLabel* state_label_ = nullptr;
    QPushButton* start_button_ = nullptr;
    QPushButton* stop_button_ = nullptr;
};

