#!/usr/bin/env python3
"""Build alternative INT8 MobileFaceNet candidates for accuracy regression."""

import argparse
from pathlib import Path

from rknn.api import RKNN


def build(onnx_path, dataset_path, output_path, algorithm):
    rknn = RKNN(verbose=True)
    try:
        ret = rknn.config(
            target_platform="rk3568",
            mean_values=[[127.5, 127.5, 127.5]],
            std_values=[[127.5, 127.5, 127.5]],
            quantized_dtype="asymmetric_quantized-8",
            quantized_algorithm=algorithm,
            quantized_method="channel",
            optimization_level=3,
        )
        if ret != 0:
            raise RuntimeError(f"config failed: {ret}")
        ret = rknn.load_onnx(
            model=str(onnx_path),
            inputs=["input.1"],
            input_size_list=[[1, 3, 112, 112]],
        )
        if ret != 0:
            raise RuntimeError(f"load_onnx failed: {ret}")
        ret = rknn.build(do_quantization=True, dataset=str(dataset_path))
        if ret != 0:
            raise RuntimeError(f"build failed: {ret}")
        ret = rknn.export_rknn(str(output_path))
        if ret != 0:
            raise RuntimeError(f"export failed: {ret}")
    finally:
        rknn.release()
    print(f"OK: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--algorithm", choices=["normal", "mmse", "kl_divergence"], required=True
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    build(args.onnx, args.dataset, args.output, args.algorithm)


if __name__ == "__main__":
    main()

