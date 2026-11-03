#include "camera_worker.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <limits>
#include <utility>
#include <vector>

#include <QMutexLocker>
#include <QPainter>
#include <QPen>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "face_engine.h"
#include "v4l2_capture.h"

using namespace std;

namespace {

const char* rotation_name(FrameRotation rotation)
{
    switch (rotation) {
    case FrameRotation::None:
        return "none";
    case FrameRotation::Left90:
        return "left";
    case FrameRotation::Right90:
        return "right";
    case FrameRotation::Rotate180:
        return "180";
    }
    return "unknown";
}

void rotate_camera_frame(cv::Mat& frame, FrameRotation rotation)
{
    // 不直接修改 V4L2/Weston 配置，只转换当前内存帧，避免影响系统桌面比例。
    cv::Mat rotated;
    switch (rotation) {
    case FrameRotation::None:
        return;
    case FrameRotation::Left90:
        cv::rotate(frame, rotated, cv::ROTATE_90_COUNTERCLOCKWISE);
        break;
    case FrameRotation::Right90:
        cv::rotate(frame, rotated, cv::ROTATE_90_CLOCKWISE);
        break;
    case FrameRotation::Rotate180:
        cv::rotate(frame, rotated, cv::ROTATE_180);
        break;
    }
    frame = std::move(rotated);
}

}  // namespace

CameraWorker::CameraWorker(const CameraWorkerConfig& config,
                           QObject* parent)
    : QThread(parent), config_(config)
{
}

CameraWorker::~CameraWorker()
{
    stop();
}

void CameraWorker::stop()
{
    // run() 循环会检查 interruption 请求；wait 让析构或“停止识别”安全收尾。
    requestInterruption();
    if (isRunning()) {
        wait(3000);
    }
}

bool CameraWorker::take_latest_frame(QImage& image)
{
    QMutexLocker locker(&mutex_);
    if (!has_new_frame_) {
        return false;
    }
    image = latest_frame_;
    has_new_frame_ = false;
    return true;
}

CameraStats CameraWorker::latest_stats() const
{
    QMutexLocker locker(&mutex_);
    return stats_;
}

void CameraWorker::run()
{
    {
        QMutexLocker locker(&mutex_);
        stats_.state = "initializing";
    }

    // RKNN context 只在本工作线程创建和销毁，避免跨线程访问 NPU 资源。
    FaceEngine engine;
    if (!engine.initialize(config_.detector_model,
                           config_.recognizer_model,
                           config_.detection_threshold,
                           config_.recognition_threshold)) {
        const QString message = QString::fromStdString(engine.last_error());
        emit errorOccurred(message);
        QMutexLocker locker(&mutex_);
        stats_.state = "model_error";
        return;
    }

    if (!engine.load_gallery(config_.gallery_directory)) {
        fprintf(stderr,
                "WARN: gallery unavailable, all faces remain unknown: %s\n",
                engine.last_error().c_str());
    }

    // 静态图模式用于界面/模型验收，不占用真实摄像头，也只需要推理第一帧。
    const bool image_mode = !config_.image_path.empty();
    cv::Mat still_image;
    V4l2Capture camera;
    if (image_mode) {
        still_image = cv::imread(config_.image_path, cv::IMREAD_COLOR);
        if (still_image.empty()) {
            emit errorOccurred(QStringLiteral("测试图片读取失败: ") +
                               QString::fromStdString(config_.image_path));
            QMutexLocker locker(&mutex_);
            stats_.state = "image_error";
            return;
        }
        const double scale = std::min(
            static_cast<double>(config_.width) / still_image.cols,
            static_cast<double>(config_.height) / still_image.rows);
        cv::Mat resized;
        cv::resize(still_image,
                   resized,
                   cv::Size(static_cast<int>(still_image.cols * scale),
                            static_cast<int>(still_image.rows * scale)),
                   0.0,
                   0.0,
                   cv::INTER_AREA);
        cv::Mat canvas(static_cast<int>(config_.height),
                       static_cast<int>(config_.width),
                       CV_8UC3,
                       cv::Scalar(15, 23, 42));
        const int offset_x = (canvas.cols - resized.cols) / 2;
        const int offset_y = (canvas.rows - resized.rows) / 2;
        resized.copyTo(canvas(cv::Rect(offset_x,
                                       offset_y,
                                       resized.cols,
                                       resized.rows)));
        still_image = canvas;
        printf("input_mode=image path=%s width=%u height=%u\n",
               config_.image_path.c_str(),
               config_.width,
               config_.height);
        fflush(stdout);
    } else if (!camera.open(config_.camera,
                            config_.width,
                            config_.height,
                            4,
                            config_.camera_fps)) {
        emit errorOccurred(QStringLiteral("摄像头打开失败: ") +
                           QString::fromStdString(config_.camera));
        QMutexLocker locker(&mutex_);
        stats_.state = "camera_error";
        return;
    }

    {
        QMutexLocker locker(&mutex_);
        stats_.state = "running";
    }

    cv::Mat frame;
    V4l2FrameMetadata metadata;
    vector<FaceResult> results;
    FaceTimings timings;
    unsigned int frame_count = 0;
    unsigned int last_inference_frame = 0;
    unsigned int inference_count = 0;
    unsigned int interval_frames = 0;
    unsigned int interval_inferences = 0;
    double video_fps = 0.0;
    double inference_fps = 0.0;
    auto rate_start = chrono::steady_clock::now();
    bool source_geometry_logged = false;

    // 采集、推理和绘制框均在后台完成；Qt 主线程只消费最终 QImage。
    while (!isInterruptionRequested()) {
        if (image_mode) {
            frame = still_image.clone();
            QThread::msleep(33);
        } else if (!camera.read(frame, metadata)) {
            emit errorOccurred(QStringLiteral("摄像头读取失败"));
            break;
        }

        if (!image_mode) {
            rotate_camera_frame(frame, config_.camera_rotation);
        }
        if (!source_geometry_logged) {
            printf("processed_input mode=%s rotation=%s width=%d height=%d\n",
                   image_mode ? "image" : "camera",
                   image_mode ? "none" : rotation_name(config_.camera_rotation),
                   frame.cols,
                   frame.rows);
            fflush(stdout);
            source_geometry_logged = true;
        }

        ++frame_count;
        ++interval_frames;
        // 实时模式每 N 帧推理一次，未推理的帧复用最近结果以保证预览流畅。
        const bool run_inference = image_mode
            ? frame_count == 1
            : (config_.inference_interval <= 1 ||
               frame_count % config_.inference_interval == 1);

        if (run_inference) {
            vector<FaceResult> fresh_results;
            FaceTimings fresh_timings;
            if (!engine.process(frame, fresh_results, fresh_timings)) {
                emit errorOccurred(QString::fromStdString(
                    "RKNN inference failed: " + engine.last_error()));
                break;
            }
            results.swap(fresh_results);
            timings = fresh_timings;
            last_inference_frame = frame_count;
            ++inference_count;
            ++interval_inferences;
        }

        string best_name = "--";
        float best_similarity = -std::numeric_limits<float>::infinity();
        for (const FaceResult& face : results) {
            const bool known = face.name != "unknown";
            const cv::Scalar color = known
                ? cv::Scalar(40, 220, 40)
                : cv::Scalar(30, 50, 240);
            cv::rectangle(frame, face.box, color, 3);
            for (const cv::Point2f& point : face.landmarks) {
                cv::circle(frame, point, 3, cv::Scalar(255, 180, 0), -1);
            }

            if (face.similarity > best_similarity) {
                best_similarity = face.similarity;
                best_name = face.name;
            }
        }

        const auto now = chrono::steady_clock::now();
        const double elapsed = chrono::duration<double>(now - rate_start).count();
        if (elapsed >= 1.0) {
            video_fps = interval_frames / elapsed;
            inference_fps = interval_inferences / elapsed;
            printf("runtime_stats frames=%u inferences=%u video_fps=%.2f "
                   "inference_fps=%.2f faces=%zu best_name=%s "
                   "best_similarity=%.3f det_ms=%.2f rec_ms=%.2f\n",
                   frame_count,
                   inference_count,
                   video_fps,
                   inference_fps,
                   results.size(),
                   best_name.c_str(),
                   std::max(0.0F, best_similarity),
                   timings.detection_ms,
                   timings.recognition_ms);
            fflush(stdout);
            interval_frames = 0;
            interval_inferences = 0;
            rate_start = now;
        }

        cv::Mat rgb;
        cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
        QImage image(rgb.data,
                     rgb.cols,
                     rgb.rows,
                     static_cast<int>(rgb.step),
                     QImage::Format_RGB888);

        {
            QPainter painter(&image);
            painter.setRenderHint(QPainter::TextAntialiasing, true);
            QFont font = painter.font();
            font.setPixelSize(25);
            font.setBold(true);
            painter.setFont(font);
            for (const FaceResult& face : results) {
                const QColor color = face.name != "unknown"
                    ? QColor(40, 220, 40)
                    : QColor(240, 50, 30);
                painter.setPen(QPen(color, 2));
                const QString display_name = face.name == "unknown"
                    ? QStringLiteral("陌生人")
                    : QString::fromUtf8(face.name.c_str());
                const QString label = QStringLiteral("%1  %2")
                    .arg(display_name)
                    .arg(std::max(0.0F, face.similarity), 0, 'f', 3);
                const int label_y = max(4, face.box.y - 34);
                painter.fillRect(QRect(face.box.x,
                                       label_y,
                                       260,
                                       32),
                                 QColor(2, 6, 23, 190));
                painter.drawText(QRect(face.box.x + 4,
                                       label_y,
                                       252,
                                       32),
                                 Qt::AlignLeft | Qt::AlignVCenter,
                                 label);
            }
        }

        {
            // 复制 QImage 后再解锁，避免 OpenCV 临时 rgb 缓冲离开本次循环后失效。
            QMutexLocker locker(&mutex_);
            latest_frame_ = image.copy();
            has_new_frame_ = true;
            stats_.video_fps = video_fps;
            stats_.inference_fps = inference_fps;
            stats_.detection_ms = timings.detection_ms;
            stats_.recognition_ms = timings.recognition_ms;
            stats_.face_count = static_cast<unsigned int>(results.size());
            stats_.result_age_frames = image_mode
                ? 0
                : frame_count - last_inference_frame;
            stats_.best_name = best_name;
            stats_.best_similarity = max(0.0F, best_similarity);
            stats_.state = "running";
        }
    }

    if (!image_mode) {
        camera.close();
    }
    engine.close();
    {
        QMutexLocker locker(&mutex_);
        stats_.state = "stopped";
    }

    printf("camera_worker_stopped frames=%u inferences=%u\n",
           frame_count,
           inference_count);
    fflush(stdout);
}

