# Third-party notices

The MIT license in this repository applies only to the original project code.

## InsightFace models

The repository does not include the InsightFace pretrained model binaries.
The project was validated with `buffalo_sc` (`det_500m.onnx` and
`w600k_mbf.onnx`). InsightFace states that its pretrained models are available
for non-commercial research only. Obtain the models and review the current
terms from the InsightFace project before use:

- https://github.com/deepinsight/insightface
- https://github.com/deepinsight/insightface/releases/tag/v0.7

## Rockchip RKNPU2 runtime

The repository does not include `rknn_api.h` or `librknnrt.so`. Obtain a
runtime that matches the RK3568 board image from Rockchip's RKNPU2 project or
the board vendor SDK, and follow the license shipped with that runtime:

- https://github.com/airockchip/rknn-toolkit2/tree/master/rknpu2

## Build dependencies

Qt, OpenCV, Linux/V4L2 and other build/runtime dependencies retain their own
licenses. They are not relicensed by this repository.
