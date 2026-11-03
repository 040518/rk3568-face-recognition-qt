#include "mainwindow.h"

#include <QApplication>
#include <QCloseEvent>
#include <QDebug>
#include <QLabel>
#include <QPixmap>
#include <QPushButton>
#include <QTimer>

MainWindow::MainWindow(const CameraWorkerConfig& config,
                       QWidget* parent)
    : QMainWindow(parent), config_(config)
{
    // 固定横屏逻辑尺寸；run_face_recognition.sh 会在运行期间将物理 MIPI 输出旋转为横屏。
    setWindowTitle(QStringLiteral("RK3568 人脸识别"));
    setWindowFlags(Qt::FramelessWindowHint | Qt::WindowStaysOnTopHint);
    setStyleSheet(QStringLiteral("QMainWindow { background:#07111f; }"));
    setFixedSize(1280, 720);

    QWidget* panel = new QWidget(this);
    setCentralWidget(panel);

    QLabel* title = new QLabel(QStringLiteral("RK3568 人脸识别"), panel);
    title->setGeometry(16, 12, 1248, 52);
    title->setAlignment(Qt::AlignCenter);
    title->setStyleSheet(
        "QLabel { color:#ffffff; background:#14213d; border-radius:10px; "
        "font-size:25px; font-weight:bold; } ");

    // 与系统桌面图标配套：关闭时触发 closeEvent，确保摄像头/NPU 线程先退出。
    QPushButton* close_button = new QPushButton(QStringLiteral("×"), panel);
    close_button->setGeometry(1192, 20, 60, 36);
    close_button->setToolTip(QStringLiteral("退出并返回桌面"));
    close_button->setStyleSheet(
        "QPushButton { background:#dc2626; color:white; border:0; "
        "border-radius:8px; font-size:27px; font-weight:bold; } "
        "QPushButton:pressed { background:#991b1b; } ");
    connect(close_button, &QPushButton::clicked,
            this, &QWidget::close);

    camera_label_ = new QLabel(QStringLiteral("摄像头未启动"), panel);
    camera_label_->setGeometry(16, 76, 350, 622);
    camera_label_->setAlignment(Qt::AlignCenter);
    camera_label_->setStyleSheet(
        "QLabel { color:#dbeafe; background:#020617; "
        "border:2px solid #3b82f6; border-radius:12px; font-size:24px; } ");

    name_label_ = make_card(QStringLiteral("姓名: --"), "#22c55e");
    name_label_->setParent(panel);
    name_label_->setGeometry(382, 76, 427, 82);

    similarity_label_ = make_card(QStringLiteral("相似度: --"), "#a855f7");
    similarity_label_->setParent(panel);
    similarity_label_->setGeometry(825, 76, 439, 82);

    fps_label_ = make_card(QStringLiteral("视频/推理 FPS: -- / --"), "#f59e0b");
    fps_label_->setParent(panel);
    fps_label_->setGeometry(382, 174, 427, 94);

    timing_label_ = make_card(QStringLiteral("检测/识别: -- / -- ms"), "#06b6d4");
    timing_label_->setParent(panel);
    timing_label_->setGeometry(825, 174, 439, 94);

    state_label_ = make_card(QStringLiteral("状态: 已停止"), "#64748b");
    state_label_->setParent(panel);
    state_label_->setGeometry(382, 284, 882, 70);

    start_button_ = new QPushButton(QStringLiteral("开始识别"), panel);
    start_button_->setGeometry(382, 370, 427, 72);
    start_button_->setStyleSheet(
        "QPushButton { background:#2563eb; color:white; border:0; "
        "border-radius:12px; font-size:23px; font-weight:bold; } "
        "QPushButton:pressed { background:#1d4ed8; } "
        "QPushButton:disabled { background:#334155; } ");

    stop_button_ = new QPushButton(QStringLiteral("停止识别"), panel);
    stop_button_->setGeometry(825, 370, 439, 72);
    stop_button_->setStyleSheet(
        "QPushButton { background:#dc2626; color:white; border:0; "
        "border-radius:12px; font-size:23px; font-weight:bold; } "
        "QPushButton:pressed { background:#b91c1c; } "
        "QPushButton:disabled { background:#334155; } ");

    QLabel* hint = new QLabel(
        QStringLiteral("绿色框：已登记\n"
                       "红色框：陌生人\n"
                       "人脸数据仅保存在本机\n"
                       "V1 未包含活体检测"), panel);
    hint->setGeometry(382, 458, 882, 240);
    hint->setAlignment(Qt::AlignCenter);
    hint->setWordWrap(true);
    hint->setStyleSheet(
        "QLabel { color:#cbd5e1; background:#0f172a; border-radius:12px; "
        "font-size:20px; padding:12px; } ");

    connect(start_button_, &QPushButton::clicked,
            this, &MainWindow::start_pipeline);
    connect(stop_button_, &QPushButton::clicked,
            this, &MainWindow::stop_pipeline);

    // UI 以约 30 FPS 刷新，不与实际摄像头 FPS 或推理 FPS 绑定。
    refresh_timer_ = new QTimer(this);
    refresh_timer_->setInterval(33);
    connect(refresh_timer_, &QTimer::timeout,
            this, &MainWindow::refresh_ui);
    refresh_timer_->start();
    set_running_state(false);

    // 窗口先显示再启动后台线程，避免首次模型加载时界面看起来无响应。
    QTimer::singleShot(800, this, &MainWindow::start_pipeline);
    if (!config_.screenshot_path.empty()) {
        QTimer::singleShot(3000, this, [this]() {
            const QString path = QString::fromStdString(config_.screenshot_path);
            const bool saved = this->grab().save(path, "PNG");
            qInfo().noquote() << "validation_screenshot=" << path
                              << "saved=" << (saved ? 1 : 0);
        });
    }
}

MainWindow::~MainWindow()
{
    stop_pipeline();
}

QLabel* MainWindow::make_card(const QString& text,
                              const QString& border_color)
{
    QLabel* label = new QLabel(text);
    label->setAlignment(Qt::AlignCenter);
    label->setStyleSheet(QString(
        "QLabel { color:#ffffff; background:rgba(15,23,42,235); "
        "border:2px solid %1; border-radius:12px; "
        "font-size:20px; font-weight:bold; padding:5px; }")
        .arg(border_color));
    return label;
}

void MainWindow::start_pipeline()
{
    // 防重复启动：桌面图标、开始按钮和自动启动可能在短时间内同时触发。
    if (worker_ != nullptr && worker_->isRunning()) {
        return;
    }
    if (worker_ != nullptr) {
        worker_->deleteLater();
        worker_ = nullptr;
    }

    camera_label_->clear();
    camera_label_->setText(config_.image_path.empty()
        ? QStringLiteral("正在初始化 RKNN 与摄像头...")
        : QStringLiteral("正在初始化 RKNN 与测试图片..."));
    state_label_->setText(QStringLiteral("状态: 初始化"));
    worker_ = new CameraWorker(config_, this);
    connect(worker_, &CameraWorker::errorOccurred,
            this, &MainWindow::show_error);
    connect(worker_, &QThread::finished, this, [this]() {
        set_running_state(false);
        if (state_label_->text() == QStringLiteral("状态: 运行中")) {
            state_label_->setText(QStringLiteral("状态: 已停止"));
        }
    });
    worker_->start();
    set_running_state(true);
}

void MainWindow::stop_pipeline()
{
    // CameraWorker::stop 会等待后台资源释放，随后才恢复按钮和占位提示。
    if (worker_ != nullptr) {
        worker_->stop();
    }
    set_running_state(false);
    camera_label_->clear();
    camera_label_->setText(QStringLiteral("输入未启动"));
    state_label_->setText(QStringLiteral("状态: 已停止"));
}

void MainWindow::refresh_ui()
{
    if (worker_ == nullptr) {
        return;
    }

    QImage image;
    // 不等待新帧；没有新帧时保留屏幕已有图像，避免 UI 被慢推理拖住。
    if (worker_->take_latest_frame(image)) {
        const QPixmap pixmap = QPixmap::fromImage(image).scaled(
            camera_label_->size(),
            Qt::KeepAspectRatio,
            Qt::FastTransformation);
        camera_label_->setPixmap(pixmap);
    }

    const CameraStats stats = worker_->latest_stats();
    if (stats.state == "running") {
        const QString source = config_.image_path.empty()
            ? QStringLiteral("实时摄像头")
            : QStringLiteral("静态图测试");
        state_label_->setText(QStringLiteral("状态: 运行中 · %1 · 人脸数: %2 · 距上次识别: %3帧")
            .arg(source)
            .arg(stats.face_count)
            .arg(stats.result_age_frames));
    }
    const QString name = stats.best_name == "unknown"
        ? QStringLiteral("陌生人")
        : QString::fromUtf8(stats.best_name.c_str());
    name_label_->setText(QStringLiteral("姓名: %1").arg(name));
    similarity_label_->setText(QStringLiteral("相似度: %1")
        .arg(stats.best_similarity, 0, 'f', 3));
    if (config_.image_path.empty()) {
        fps_label_->setText(QStringLiteral("视频/推理 FPS\n%1 / %2")
            .arg(stats.video_fps, 0, 'f', 1)
            .arg(stats.inference_fps, 0, 'f', 1));
    } else {
        fps_label_->setText(QStringLiteral("输入/推理模式\n静态图 / 单次"));
    }
    timing_label_->setText(QStringLiteral("检测/识别 ms\n%1 / %2")
        .arg(stats.detection_ms, 0, 'f', 1)
        .arg(stats.recognition_ms, 0, 'f', 1));
}

void MainWindow::show_error(const QString& message)
{
    state_label_->setText(QStringLiteral("错误: ") + message);
    camera_label_->setText(message);
    set_running_state(false);
}

void MainWindow::set_running_state(bool running)
{
    start_button_->setEnabled(!running);
    stop_button_->setEnabled(running);
}

void MainWindow::closeEvent(QCloseEvent* event)
{
    // 退出按钮、窗口管理器关闭和异常关闭均走同一条释放路径。
    stop_pipeline();
    event->accept();
}

