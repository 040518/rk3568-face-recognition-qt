# 模型文件

此目录不提交第三方模型二进制。请从 InsightFace GitHub Release 下载
`buffalo_sc.zip`，提取 ONNX 文件后再使用 `tools/convert_models.py` 和
`tools/tune_recognizer.py` 转换。`checksums.sha256` 记录了本项目验证过的文件哈希。

- `original/det_500m.onnx`
- `original/w600k_mbf.onnx`
- `rknn/det_500m_rk3568_fp16.rknn`
- `rknn/det_500m_rk3568_int8.rknn`
- `rknn/w600k_mbf_rk3568_fp16.rknn`
- `rknn/w600k_mbf_rk3568_int8.rknn`
- `rknn/w600k_mbf_rk3568_int8_mmse.rknn`（回归通过，默认）

预训练权重只允许非商业研究使用。INT8 文件只有通过
`docs/quantization-report.md` 中的量化回归标准后才能设为默认模型。

