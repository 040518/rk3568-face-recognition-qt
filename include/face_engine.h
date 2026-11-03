#pragma once

#include <array>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "rknn_api.h"

// 一张脸的最终可显示结果：检测框、五个关键点和身份比对结果。
struct FaceResult
{
    cv::Rect box;
    std::array<cv::Point2f, 5> landmarks{};
    float detection_score = 0.0F;
    std::string name = "unknown";
    float similarity = 0.0F;
};

// 一次推理中检测、识别两个阶段的耗时，单位为毫秒。
struct FaceTimings
{
    double detection_ms = 0.0;
    double recognition_ms = 0.0;
};

// RKNN 人脸算法封装：SCRFD 检测 -> 五点对齐 -> MobileFaceNet 特征 -> 余弦比对。
// 一个实例只在 CameraWorker 工作线程中使用，因此不需要在此类内部额外加锁。
class FaceEngine
{
public:
    FaceEngine();
    ~FaceEngine();

    FaceEngine(const FaceEngine&) = delete;
    FaceEngine& operator=(const FaceEngine&) = delete;

    bool initialize(const std::string& detector_model,
                    const std::string& recognizer_model,
                    float detection_threshold,
                    float recognition_threshold);

    bool load_gallery(const std::string& gallery_directory);

    bool process(const cv::Mat& bgr_frame,
                 std::vector<FaceResult>& results,
                 FaceTimings& timings);

    void close();
    const std::string& last_error() const;

private:
    // 一个 RKNN 模型的上下文和模型输入/输出元数据。
    struct Model
    {
        rknn_context context = 0;
        int width = 0;
        int height = 0;
        int channels = 0;
        unsigned int output_count = 0;
        bool initialized = false;
    };

    // SCRFD 解码后的候选脸，尚未经过身份特征提取。
    struct Candidate
    {
        cv::Rect2f box;
        std::array<cv::Point2f, 5> landmarks{};
        float score = 0.0F;
    };

    // 人脸库中一个姓名对应的归一化平均特征向量。
    struct Identity
    {
        std::string name;
        std::vector<float> embedding;
    };

    bool initialize_model(const std::string& path, Model& model);
    // 输入必须是模型尺寸的连续 RGB uint8 三通道图像；输出统一转为 float 方便后处理。
    bool infer(Model& model,
               const cv::Mat& rgb_image,
               std::vector<std::vector<float>>& outputs);

    bool detect(const cv::Mat& bgr_frame,
                std::vector<Candidate>& candidates);
    bool embed(const cv::Mat& aligned_bgr,
               std::vector<float>& embedding);
    bool align_face(const cv::Mat& bgr_frame,
                    const std::array<cv::Point2f, 5>& landmarks,
                    cv::Mat& aligned);

    void decode_scrfd(const std::vector<std::vector<float>>& outputs,
                      float scale,
                      float pad_x,
                      float pad_y,
                      const cv::Size& source_size,
                      std::vector<Candidate>& candidates) const;

    // 在已加载的人脸库中取最高余弦相似度；低于阈值时返回 unknown。
    std::pair<std::string, float> match(
        const std::vector<float>& embedding) const;

    bool build_identity(const std::string& name,
                        const std::string& directory,
                        Identity& identity);

    static void normalize(std::vector<float>& values);
    static float intersection_over_union(const cv::Rect2f& a,
                                         const cv::Rect2f& b);

    Model detector_;
    Model recognizer_;
    std::vector<Identity> gallery_;
    float detection_threshold_ = 0.5F;
    float recognition_threshold_ = 0.45F;
    std::string last_error_;
};

