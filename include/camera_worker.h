#pragma once

#include <string>

#include <QImage>
#include <QMutex>
#include <QThread>

// 摄像头画面在进入检测、识别和 Qt 绘制前的旋转方式。
// 注意：这只旋转应用内的图像，不修改系统桌面方向。
enum class FrameRotation
{
    None,
    Left90,
    Right90,
    Rotate180,
};

// 主线程传给工作线程的启动参数。实时摄像头和静态图验收共用此配置。
struct CameraWorkerConfig
{
    std::string camera = "/dev/video0";
    unsigned int width = 1280;
    unsigned int height = 720;
    unsigned int camera_fps = 0;
    FrameRotation camera_rotation = FrameRotation::Left90;
    std::string image_path;
    std::string screenshot_path;
    std::string detector_model;
    std::string recognizer_model;
    std::string gallery_directory;
    float detection_threshold = 0.5F;
    float recognition_threshold = 0.45F;
    unsigned int inference_interval = 2;
};

// 工作线程每帧更新的只读状态；Qt 主线程只读取副本，避免直接操作摄像头对象。
struct CameraStats
{
    double video_fps = 0.0;
    double inference_fps = 0.0;
    double detection_ms = 0.0;
    double recognition_ms = 0.0;
    unsigned int face_count = 0;
    unsigned int result_age_frames = 0;
    std::string best_name = "--";
    float best_similarity = 0.0F;
    std::string state = "stopped";
};

// 后台线程：负责 V4L2 采集、RKNN 推理、叠加检测结果，并只保存最新一帧。
// “只保存最新一帧”可防止 GUI 一时变慢时出现画面越积越旧的情况。
class CameraWorker : public QThread
{
    Q_OBJECT

public:
    explicit CameraWorker(const CameraWorkerConfig& config,
                          QObject* parent = nullptr);
    ~CameraWorker() override;

    // 成功时取走一帧并清除“有新帧”标记；GUI 下一次刷新只显示更新后的画面。
    bool take_latest_frame(QImage& image);
    CameraStats latest_stats() const;
    // 请求线程退出，并最多等待 3 秒让 V4L2/RKNN 正常释放资源。
    void stop();

signals:
    void errorOccurred(const QString& message);

protected:
    void run() override;

private:
    CameraWorkerConfig config_;
    mutable QMutex mutex_;
    QImage latest_frame_;
    bool has_new_frame_ = false;
    CameraStats stats_;
};

