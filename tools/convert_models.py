#!/usr/bin/env python3
"""Convert the fixed InsightFace pair to RK3568 FP16 and calibrated INT8."""

import argparse
from pathlib import Path

from rknn.api import RKNN


MODELS = {
    "detector": {
        "input": [1, 3, 640, 640],
        "mean": [[127.5, 127.5, 127.5]],
        "std": [[128.0, 128.0, 128.0]],
        "stem": "det_500m_rk3568",
    },
    "recognizer": {
        "input": [1, 3, 112, 112],
        "mean": [[127.5, 127.5, 127.5]],
        "std": [[127.5, 127.5, 127.5]],
        "stem": "w600k_mbf_rk3568",
    },
}


def convert(kind, onnx_path, dataset_path, output_dir, quantized):
    spec = MODELS[kind]
    suffix = "int8" if quantized else "fp16"
    output_path = output_dir / f"{spec['stem']}_{suffix}.rknn"
    rknn = RKNN(verbose=True)
    try:
        config = dict(
            target_platform="rk3568",
            mean_values=spec["mean"],
            std_values=spec["std"],
        )
        if quantized:
            config.update(
                quantized_dtype="asymmetric_quantized-8",
                quantized_algorithm="normal",
                quantized_method="channel",
            )
        ret = rknn.config(**config)
        if ret != 0:
            raise RuntimeError(f"rknn.config failed: {ret}")

        ret = rknn.load_onnx(
            model=str(onnx_path),
            inputs=["input.1"],
            input_size_list=[spec["input"]],
        )
        if ret != 0:
            raise RuntimeError(f"rknn.load_onnx failed: {ret}")

        ret = rknn.build(
            do_quantization=quantized,
            dataset=str(dataset_path) if quantized else None,
        )
        if ret != 0:
            raise RuntimeError(f"rknn.build failed: {ret}")

        ret = rknn.export_rknn(str(output_path))
        if ret != 0:
            raise RuntimeError(f"rknn.export_rknn failed: {ret}")
    finally:
        rknn.release()

    print(f"OK: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--det", type=Path, required=True)
    parser.add_argument("--rec", type=Path, required=True)
    parser.add_argument("--det-dataset", type=Path, required=True)
    parser.add_argument("--rec-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for path in (args.det, args.rec, args.det_dataset, args.rec_dataset):
        if not path.is_file():
            parser.error(f"missing file: {path}")
    args.output.mkdir(parents=True, exist_ok=True)

    convert("detector", args.det, args.det_dataset,
            args.output, quantized=False)
    convert("detector", args.det, args.det_dataset,
            args.output, quantized=True)
    convert("recognizer", args.rec, args.rec_dataset,
            args.output, quantized=False)
    convert("recognizer", args.rec, args.rec_dataset,
            args.output, quantized=True)


if __name__ == "__main__":
    main()

