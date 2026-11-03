#include <QApplication>
#include <QCommandLineOption>
#include <QCommandLineParser>
#include <QDir>
#include <QScreen>
#include <QTextStream>

#include "mainwindow.h"

int main(int argc, char* argv[])
{
    // QApplication 必须在最前面创建，后续命令行参数解析和 MIPI 屏幕查询都依赖它。
    QApplication app(argc, argv);
    QApplication::setApplicationName(QStringLiteral("rk3568-face-recognition"));
    if (app.primaryScreen() != nullptr) {
        const QSize screen_size = app.primaryScreen()->size();
        QTextStream(stdout) << "qt_screen=" << screen_size.width()
                            << "x" << screen_size.height() << "\n";
    }

    // 所有运行时可调整项都集中在命令行中，默认值对应已验证的板端配置。
    QCommandLineParser parser;
    parser.setApplicationDescription(
        QStringLiteral("RK3568 RKNN 人脸识别与 MIPI 显示"));
    parser.addHelpOption();

    const QString base = QApplication::applicationDirPath();
    QCommandLineOption detector(
        QStringList() << "d" << "det-model",
        QStringLiteral("SCRFD RKNN model"),
        QStringLiteral("path"),
        base + "/models/det_500m_rk3568_int8.rknn");
    QCommandLineOption recognizer(
        QStringList() << "r" << "rec-model",
        QStringLiteral("MobileFaceNet RKNN model"),
        QStringLiteral("path"),
        base + "/models/w600k_mbf_rk3568_int8_mmse.rknn");
    QCommandLineOption gallery(
        QStringList() << "g" << "gallery",
        QStringLiteral("Authorized gallery directory"),
        QStringLiteral("path"),
        base + "/gallery");
    QCommandLineOption camera(
        QStringList() << "c" << "camera",
        QStringLiteral("V4L2 camera device"),
        QStringLiteral("device"),
        "/dev/video0");
    QCommandLineOption image(
        QStringList() << "image",
        QStringLiteral("Use one image repeatedly instead of the camera"),
        QStringLiteral("path"));
    QCommandLineOption screenshot(
        QStringList() << "screenshot",
        QStringLiteral("Save one Qt validation screenshot (requires --image)"),
        QStringLiteral("path"));
    QCommandLineOption threshold(
        QStringList() << "t" << "threshold",
        QStringLiteral("Cosine recognition threshold"),
        QStringLiteral("value"),
        "0.45");
    QCommandLineOption interval(
        QStringList() << "i" << "interval",
        QStringLiteral("Run inference every N frames"),
        QStringLiteral("frames"),
        "2");
    QCommandLineOption camera_fps(
        QStringList() << "camera-fps",
        QStringLiteral("Requested V4L2 capture FPS; 0 keeps sensor mode"),
        QStringLiteral("fps"),
        "0");
    QCommandLineOption camera_rotation(
        QStringList() << "camera-rotation",
        QStringLiteral("Rotate live camera before inference: left, right, 180, none"),
        QStringLiteral("rotation"),
        "left");

    parser.addOptions({detector, recognizer, gallery,
                       camera, image, screenshot,
                       threshold, interval, camera_fps, camera_rotation});
    parser.process(app);

    bool threshold_ok = false;
    bool interval_ok = false;
    bool camera_fps_ok = false;
    const float threshold_value = parser.value(threshold).toFloat(&threshold_ok);
    const unsigned int interval_value = parser.value(interval).toUInt(&interval_ok);
    const unsigned int camera_fps_value =
        parser.value(camera_fps).toUInt(&camera_fps_ok);
    if (!threshold_ok || threshold_value < -1.0F || threshold_value > 1.0F ||
        !interval_ok || interval_value == 0 ||
        !camera_fps_ok || camera_fps_value > 240) {
        QTextStream(stderr) << "Invalid threshold, interval, or camera FPS\n";
        return 2;
    }
    // 实时摄像头画面可能包含未授权人脸，禁止在该模式下导出截图。
    if (parser.isSet(screenshot) && !parser.isSet(image)) {
        QTextStream(stderr)
            << "--screenshot is allowed only with --image validation mode\n";
        return 2;
    }

    // MIPI 安装方向决定实时画面需要向左转 90°；仅作用于应用输入画面。
    FrameRotation rotation_value = FrameRotation::Left90;
    const QString rotation_text = parser.value(camera_rotation).toLower();
    if (rotation_text == QStringLiteral("none")) {
        rotation_value = FrameRotation::None;
    } else if (rotation_text == QStringLiteral("left")) {
        rotation_value = FrameRotation::Left90;
    } else if (rotation_text == QStringLiteral("right")) {
        rotation_value = FrameRotation::Right90;
    } else if (rotation_text == QStringLiteral("180")) {
        rotation_value = FrameRotation::Rotate180;
    } else {
        QTextStream(stderr)
            << "Invalid --camera-rotation; use left, right, 180, or none\n";
        return 2;
    }

    // 将命令行文本转换为强类型配置，之后只把 config 交给 Qt 窗口和工作线程。
    CameraWorkerConfig config;
    config.camera = parser.value(camera).toStdString();
    config.image_path = parser.value(image).toStdString();
    config.screenshot_path = parser.value(screenshot).toStdString();
    config.camera_fps = camera_fps_value;
    config.camera_rotation = rotation_value;
    config.detector_model = parser.value(detector).toStdString();
    config.recognizer_model = parser.value(recognizer).toStdString();
    config.gallery_directory = parser.value(gallery).toStdString();
    config.recognition_threshold = threshold_value;
    config.inference_interval = interval_value;

    // 界面始终使用横向 1280×720 逻辑坐标；启动脚本负责临时旋转 Weston 输出。
    MainWindow window(config);
    window.showFullScreen();
    return app.exec();
}

