#include "face_engine.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <numeric>
#include <sys/stat.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;

namespace
{
bool read_binary(const string& path, vector<unsigned char>& bytes)
{
    ifstream input(path, ios::binary | ios::ate);
    if (!input) {
        return false;
    }

    const streamsize length = input.tellg();
    if (length <= 0) {
        return false;
    }

    bytes.resize(static_cast<size_t>(length));
    input.seekg(0, ios::beg);
    return static_cast<bool>(input.read(
        reinterpret_cast<char*>(bytes.data()), length));
}

bool is_directory(const string& path)
{
    struct stat info{};
    return stat(path.c_str(), &info) == 0 && S_ISDIR(info.st_mode);
}

bool is_image_name(const string& name)
{
    string lower = name;
    transform(lower.begin(), lower.end(), lower.begin(),
              [](unsigned char value) {
                  return static_cast<char>(tolower(value));
              });

    const array<string, 4> suffixes = {".jpg", ".jpeg", ".png", ".bmp"};
    for (const string& suffix : suffixes) {
        if (lower.size() >= suffix.size() &&
            lower.compare(lower.size() - suffix.size(),
                          suffix.size(), suffix) == 0) {
            return true;
        }
    }
    return false;
}
}

FaceEngine::FaceEngine() = default;

FaceEngine::~FaceEngine()
{
    close();
}

bool FaceEngine::initialize(const string& detector_model,
                            const string& recognizer_model,
                            float detection_threshold,
                            float recognition_threshold)
{
    // 支持重复初始化：先清理旧 RKNN context，再加载检测和识别两个模型。
    close();
    detection_threshold_ = detection_threshold;
    recognition_threshold_ = recognition_threshold;

    if (!initialize_model(detector_model, detector_)) {
        return false;
    }
    if (!initialize_model(recognizer_model, recognizer_)) {
        close();
        return false;
    }

    if (detector_.output_count != 9) {
        last_error_ = "SCRFD detector must expose 9 outputs; got " +
                      to_string(detector_.output_count);
        close();
        return false;
    }
    if (recognizer_.output_count < 1) {
        last_error_ = "recognizer has no output";
        close();
        return false;
    }

    printf("face_engine detector=%dx%d outputs=%u "
           "recognizer=%dx%d outputs=%u\n",
           detector_.width, detector_.height, detector_.output_count,
           recognizer_.width, recognizer_.height, recognizer_.output_count);
    fflush(stdout);
    return true;
}

bool FaceEngine::initialize_model(const string& path, Model& model)
{
    // RKNN Runtime 从内存加载 .rknn；随后查询真实输入形状，避免把模型尺寸写死。
    vector<unsigned char> bytes;
    if (!read_binary(path, bytes)) {
        last_error_ = "cannot read model: " + path;
        return false;
    }

    int rc = rknn_init(&model.context,
                       bytes.data(),
                       static_cast<unsigned int>(bytes.size()),
                       0,
                       nullptr);
    if (rc != RKNN_SUCC) {
        last_error_ = "rknn_init failed: " + path + " rc=" + to_string(rc);
        return false;
    }

    rknn_input_output_num count{};
    rc = rknn_query(model.context,
                    RKNN_QUERY_IN_OUT_NUM,
                    &count,
                    sizeof(count));
    if (rc != RKNN_SUCC || count.n_input != 1) {
        last_error_ = "model must have exactly one input: " + path;
        rknn_destroy(model.context);
        model.context = 0;
        return false;
    }

    rknn_tensor_attr input{};
    input.index = 0;
    rc = rknn_query(model.context,
                    RKNN_QUERY_INPUT_ATTR,
                    &input,
                    sizeof(input));
    if (rc != RKNN_SUCC || input.n_dims != 4) {
        last_error_ = "cannot query 4D input: " + path;
        rknn_destroy(model.context);
        model.context = 0;
        return false;
    }

    if (input.fmt == RKNN_TENSOR_NCHW) {
        model.channels = static_cast<int>(input.dims[1]);
        model.height = static_cast<int>(input.dims[2]);
        model.width = static_cast<int>(input.dims[3]);
    } else {
        model.height = static_cast<int>(input.dims[1]);
        model.width = static_cast<int>(input.dims[2]);
        model.channels = static_cast<int>(input.dims[3]);
    }

    if (model.width <= 0 || model.height <= 0 || model.channels != 3) {
        last_error_ = "unsupported model input shape: " + path;
        rknn_destroy(model.context);
        model.context = 0;
        return false;
    }

    model.output_count = count.n_output;
    model.initialized = true;
    return true;
}

bool FaceEngine::infer(Model& model,
                       const cv::Mat& rgb_image,
                       vector<vector<float>>& outputs)
{
    // 两个模型均使用 uint8、NHWC、RGB 输入；输出请求 Runtime 转 float 便于统一解码。
    if (!model.initialized || rgb_image.empty() ||
        rgb_image.cols != model.width || rgb_image.rows != model.height ||
        rgb_image.type() != CV_8UC3 || !rgb_image.isContinuous()) {
        last_error_ = "invalid RKNN input image";
        return false;
    }

    rknn_input input{};
    input.index = 0;
    input.type = RKNN_TENSOR_UINT8;
    input.fmt = RKNN_TENSOR_NHWC;
    input.size = static_cast<unsigned int>(rgb_image.total() * 3);
    input.buf = const_cast<unsigned char*>(rgb_image.ptr<unsigned char>());
    input.pass_through = 0;

    int rc = rknn_inputs_set(model.context, 1, &input);
    if (rc != RKNN_SUCC) {
        last_error_ = "rknn_inputs_set failed rc=" + to_string(rc);
        return false;
    }

    rc = rknn_run(model.context, nullptr);
    if (rc != RKNN_SUCC) {
        last_error_ = "rknn_run failed rc=" + to_string(rc);
        return false;
    }

    vector<rknn_output> raw(model.output_count);
    for (unsigned int index = 0; index < model.output_count; ++index) {
        raw[index].index = index;
        raw[index].want_float = 1;
    }

    rc = rknn_outputs_get(model.context,
                          model.output_count,
                          raw.data(),
                          nullptr);
    if (rc != RKNN_SUCC) {
        last_error_ = "rknn_outputs_get failed rc=" + to_string(rc);
        return false;
    }

    outputs.clear();
    outputs.resize(model.output_count);
    for (unsigned int index = 0; index < model.output_count; ++index) {
        const size_t count = raw[index].size / sizeof(float);
        const float* source = static_cast<const float*>(raw[index].buf);
        outputs[index].assign(source, source + count);
    }

    rknn_outputs_release(model.context,
                         model.output_count,
                         raw.data());
    return true;
}

bool FaceEngine::detect(const cv::Mat& bgr_frame,
                        vector<Candidate>& candidates)
{
    // 采用等比例缩放 + 灰边(letterbox)，并把 scale/pad 传给解码器还原到原图坐标。
    const float scale = min(
        static_cast<float>(detector_.width) / bgr_frame.cols,
        static_cast<float>(detector_.height) / bgr_frame.rows);
    const int resized_width = max(1, static_cast<int>(round(bgr_frame.cols * scale)));
    const int resized_height = max(1, static_cast<int>(round(bgr_frame.rows * scale)));
    const int pad_x = (detector_.width - resized_width) / 2;
    const int pad_y = (detector_.height - resized_height) / 2;

    cv::Mat resized;
    cv::resize(bgr_frame,
               resized,
               cv::Size(resized_width, resized_height),
               0.0,
               0.0,
               cv::INTER_LINEAR);

    cv::Mat canvas(detector_.height,
                   detector_.width,
                   CV_8UC3,
                   cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(pad_x,
                                   pad_y,
                                   resized_width,
                                   resized_height)));

    cv::Mat rgb;
    cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);
    if (!rgb.isContinuous()) {
        rgb = rgb.clone();
    }

    vector<vector<float>> outputs;
    if (!infer(detector_, rgb, outputs)) {
        return false;
    }

    decode_scrfd(outputs,
                 scale,
                 static_cast<float>(pad_x),
                 static_cast<float>(pad_y),
                 bgr_frame.size(),
                 candidates);
    return true;
}

void FaceEngine::decode_scrfd(const vector<vector<float>>& outputs,
                              float scale,
                              float pad_x,
                              float pad_y,
                              const cv::Size& source_size,
                              vector<Candidate>& candidates) const
{
    // SCRFD 的 9 个输出依次是 3 个尺度的 score、box 和 5 点关键点。
    vector<Candidate> decoded;
    const array<int, 3> strides = {8, 16, 32};

    for (size_t level = 0; level < strides.size(); ++level) {
        const int stride = strides[level];
        const int feature_width = detector_.width / stride;
        const int feature_height = detector_.height / stride;
        const size_t spatial = static_cast<size_t>(feature_width) *
                               static_cast<size_t>(feature_height);

        const vector<float>& scores = outputs[level];
        const vector<float>& boxes = outputs[level + 3];
        const vector<float>& points = outputs[level + 6];
        if (spatial == 0 || scores.empty()) {
            continue;
        }

        const size_t score_anchors = scores.size() / spatial;
        const size_t box_anchors = boxes.size() / (spatial * 4);
        const size_t point_anchors = points.size() / (spatial * 10);
        const size_t anchors = min(score_anchors,
                                   min(box_anchors, point_anchors));
        if (anchors == 0) {
            continue;
        }

        for (int y = 0; y < feature_height; ++y) {
            for (int x = 0; x < feature_width; ++x) {
                for (size_t anchor = 0; anchor < anchors; ++anchor) {
                    const size_t index =
                        (static_cast<size_t>(y) * feature_width + x) *
                        anchors + anchor;
                    const float score = scores[index];
                    if (score < detection_threshold_) {
                        continue;
                    }

                    const float center_x = static_cast<float>(x * stride);
                    const float center_y = static_cast<float>(y * stride);
                    const size_t box_base = index * 4;
                    float left = center_x - boxes[box_base] * stride;
                    float top = center_y - boxes[box_base + 1] * stride;
                    float right = center_x + boxes[box_base + 2] * stride;
                    float bottom = center_y + boxes[box_base + 3] * stride;

                    left = (left - pad_x) / scale;
                    top = (top - pad_y) / scale;
                    right = (right - pad_x) / scale;
                    bottom = (bottom - pad_y) / scale;

                    left = max(0.0F, min(left,
                        static_cast<float>(source_size.width - 1)));
                    top = max(0.0F, min(top,
                        static_cast<float>(source_size.height - 1)));
                    right = max(0.0F, min(right,
                        static_cast<float>(source_size.width - 1)));
                    bottom = max(0.0F, min(bottom,
                        static_cast<float>(source_size.height - 1)));
                    if (right - left < 2.0F || bottom - top < 2.0F) {
                        continue;
                    }

                    Candidate candidate;
                    candidate.box = cv::Rect2f(left,
                                               top,
                                               right - left,
                                               bottom - top);
                    candidate.score = score;
                    const size_t point_base = index * 10;
                    for (size_t point = 0; point < 5; ++point) {
                        const float px = center_x +
                            points[point_base + point * 2] * stride;
                        const float py = center_y +
                            points[point_base + point * 2 + 1] * stride;
                        candidate.landmarks[point] = cv::Point2f(
                            (px - pad_x) / scale,
                            (py - pad_y) / scale);
                    }
                    decoded.push_back(candidate);
                }
            }
        }
    }

    sort(decoded.begin(), decoded.end(),
         [](const Candidate& left, const Candidate& right) {
             return left.score > right.score;
         });

    // 按置信度从高到低做 NMS，IoU 超过 0.4 的重复框只保留一个。
    candidates.clear();
    for (const Candidate& candidate : decoded) {
        bool suppressed = false;
        for (const Candidate& kept : candidates) {
            if (intersection_over_union(candidate.box, kept.box) > 0.4F) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) {
            candidates.push_back(candidate);
        }
    }
}

bool FaceEngine::align_face(
    const cv::Mat& bgr_frame,
    const array<cv::Point2f, 5>& landmarks,
    cv::Mat& aligned)
{
    // 将检测到的五点仿射变换到 ArcFace 标准模板，降低姿态差异对特征的影响。
    const float scale_x = recognizer_.width / 112.0F;
    const float scale_y = recognizer_.height / 112.0F;
    const array<cv::Point2f, 5> canonical = {
        cv::Point2f(38.2946F * scale_x, 51.6963F * scale_y),
        cv::Point2f(73.5318F * scale_x, 51.5014F * scale_y),
        cv::Point2f(56.0252F * scale_x, 71.7366F * scale_y),
        cv::Point2f(41.5493F * scale_x, 92.3655F * scale_y),
        cv::Point2f(70.7299F * scale_x, 92.2041F * scale_y)
    };

    vector<cv::Point2f> source(landmarks.begin(), landmarks.end());
    vector<cv::Point2f> target(canonical.begin(), canonical.end());
    cv::Mat inliers;
    const cv::Mat transform = cv::estimateAffinePartial2D(
        source, target, inliers, cv::LMEDS);
    if (transform.empty()) {
        last_error_ = "five-point face alignment failed";
        return false;
    }

    cv::warpAffine(bgr_frame,
                   aligned,
                   transform,
                   cv::Size(recognizer_.width, recognizer_.height),
                   cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT,
                   cv::Scalar(0, 0, 0));
    return !aligned.empty();
}

bool FaceEngine::embed(const cv::Mat& aligned_bgr,
                       vector<float>& embedding)
{
    cv::Mat rgb;
    cv::cvtColor(aligned_bgr, rgb, cv::COLOR_BGR2RGB);
    if (!rgb.isContinuous()) {
        rgb = rgb.clone();
    }

    vector<vector<float>> outputs;
    if (!infer(recognizer_, rgb, outputs) || outputs.empty()) {
        return false;
    }

    embedding = outputs.front();
    normalize(embedding);
    return !embedding.empty();
}

bool FaceEngine::process(const cv::Mat& bgr_frame,
                         vector<FaceResult>& results,
                         FaceTimings& timings)
{
    // 一次完整处理对应“检测 -> 对齐/特征 -> 比对”，并分别记录检测和识别耗时。
    results.clear();
    timings = FaceTimings{};
    if (bgr_frame.empty() || bgr_frame.type() != CV_8UC3) {
        last_error_ = "process requires non-empty BGR CV_8UC3 frame";
        return false;
    }

    const auto detection_start = chrono::steady_clock::now();
    vector<Candidate> candidates;
    if (!detect(bgr_frame, candidates)) {
        return false;
    }
    timings.detection_ms = chrono::duration<double, milli>(
        chrono::steady_clock::now() - detection_start).count();

    const auto recognition_start = chrono::steady_clock::now();
    for (const Candidate& candidate : candidates) {
        cv::Mat aligned;
        if (!align_face(bgr_frame, candidate.landmarks, aligned)) {
            continue;
        }

        vector<float> embedding;
        if (!embed(aligned, embedding)) {
            continue;
        }

        const pair<string, float> identity = match(embedding);
        FaceResult result;
        result.box = cv::Rect(
            static_cast<int>(round(candidate.box.x)),
            static_cast<int>(round(candidate.box.y)),
            static_cast<int>(round(candidate.box.width)),
            static_cast<int>(round(candidate.box.height)));
        result.landmarks = candidate.landmarks;
        result.detection_score = candidate.score;
        result.name = identity.first;
        result.similarity = identity.second;
        results.push_back(result);
    }
    timings.recognition_ms = chrono::duration<double, milli>(
        chrono::steady_clock::now() - recognition_start).count();
    return true;
}

bool FaceEngine::load_gallery(const string& gallery_directory)
{
    // gallery/<姓名>/*.jpg：每个姓名的多张特征会在 build_identity 中取归一化均值。
    gallery_.clear();
    DIR* root = opendir(gallery_directory.c_str());
    if (root == nullptr) {
        last_error_ = "cannot open gallery directory: " + gallery_directory;
        return false;
    }

    dirent* entry = nullptr;
    while ((entry = readdir(root)) != nullptr) {
        const string name(entry->d_name);
        if (name == "." || name == "..") {
            continue;
        }
        const string path = gallery_directory + "/" + name;
        if (!is_directory(path)) {
            continue;
        }

        Identity identity;
        if (build_identity(name, path, identity)) {
            printf("gallery_loaded name=%s dims=%zu\n",
                   identity.name.c_str(), identity.embedding.size());
            gallery_.push_back(move(identity));
        } else {
            fprintf(stderr,
                    "WARN: gallery identity skipped name=%s reason=%s\n",
                    name.c_str(), last_error_.c_str());
        }
    }
    closedir(root);

    printf("gallery_identities=%zu\n", gallery_.size());
    fflush(stdout);
    return true;
}

bool FaceEngine::build_identity(const string& name,
                                const string& directory,
                                Identity& identity)
{
    DIR* folder = opendir(directory.c_str());
    if (folder == nullptr) {
        last_error_ = "cannot open identity directory";
        return false;
    }

    vector<vector<float>> samples;
    dirent* entry = nullptr;
    while ((entry = readdir(folder)) != nullptr) {
        const string file_name(entry->d_name);
        if (!is_image_name(file_name)) {
            continue;
        }

        const string path = directory + "/" + file_name;
        cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
        if (image.empty()) {
            continue;
        }

        vector<Candidate> faces;
        if (!detect(image, faces) || faces.empty()) {
            continue;
        }

        // 一张登记照可能误检出多张脸；默认使用面积最大的脸构建该人的特征样本。
        const auto largest = max_element(
            faces.begin(), faces.end(),
            [](const Candidate& left, const Candidate& right) {
                return left.box.area() < right.box.area();
            });

        cv::Mat aligned;
        vector<float> embedding;
        if (align_face(image, largest->landmarks, aligned) &&
            embed(aligned, embedding)) {
            samples.push_back(move(embedding));
        }
    }
    closedir(folder);

    if (samples.empty()) {
        last_error_ = "no usable face image";
        return false;
    }

    identity.name = name;
    identity.embedding.assign(samples.front().size(), 0.0F);
    for (const vector<float>& sample : samples) {
        if (sample.size() != identity.embedding.size()) {
            continue;
        }
        for (size_t index = 0; index < sample.size(); ++index) {
            identity.embedding[index] += sample[index];
        }
    }
    normalize(identity.embedding);
    return true;
}

pair<string, float> FaceEngine::match(
    const vector<float>& embedding) const
{
    // 特征都已 L2 归一化，点积即余弦相似度。
    string best_name = "unknown";
    float best_score = -1.0F;
    for (const Identity& identity : gallery_) {
        if (identity.embedding.size() != embedding.size()) {
            continue;
        }
        const float score = inner_product(
            embedding.begin(), embedding.end(),
            identity.embedding.begin(), 0.0F);
        if (score > best_score) {
            best_score = score;
            best_name = identity.name;
        }
    }

    if (best_score < recognition_threshold_) {
        best_name = "unknown";
    }
    return make_pair(best_name, best_score);
}

void FaceEngine::normalize(vector<float>& values)
{
    double squared_sum = 0.0;
    for (float value : values) {
        squared_sum += static_cast<double>(value) * value;
    }
    const float length = static_cast<float>(sqrt(squared_sum));
    if (length <= 1e-12F) {
        return;
    }
    for (float& value : values) {
        value /= length;
    }
}

float FaceEngine::intersection_over_union(const cv::Rect2f& a,
                                          const cv::Rect2f& b)
{
    const cv::Rect2f overlap = a & b;
    const float intersection = max(0.0F, overlap.area());
    const float united = a.area() + b.area() - intersection;
    return united > 0.0F ? intersection / united : 0.0F;
}

void FaceEngine::close()
{
    // 销毁顺序与初始化相反；重复调用安全，便于错误路径和析构函数复用。
    gallery_.clear();
    if (recognizer_.initialized) {
        rknn_destroy(recognizer_.context);
    }
    if (detector_.initialized) {
        rknn_destroy(detector_.context);
    }
    detector_ = Model{};
    recognizer_ = Model{};
}

const string& FaceEngine::last_error() const
{
    return last_error_;
}

