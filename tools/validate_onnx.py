#!/usr/bin/env python3
"""Validate the CPU ONNX face-recognition baseline on aligned images."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def identity_key(path: Path) -> str:
    parts = path.stem.split("_")
    return "_".join(parts[1:3]) if len(parts) >= 4 else path.stem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    args = parser.parse_args()

    image_paths = sorted(args.images.glob("*.jpg"))
    if not image_paths:
        parser.error("no JPG images found")

    session = ort.InferenceSession(
        str(args.model), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    embeddings = []
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            raise SystemExit(f"cannot read image: {path}")
        image = cv2.resize(image, (112, 112))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        tensor = ((rgb - 127.5) / 127.5).transpose(2, 0, 1)[None]
        embedding = session.run(None, {input_name: tensor})[0].reshape(-1)
        norm = float(np.linalg.norm(embedding))
        if not np.isfinite(embedding).all() or norm <= 1e-12:
            raise SystemExit(f"invalid embedding: {path}")
        embeddings.append(embedding / norm)

    same_scores = []
    different_scores = []
    for left in range(len(image_paths)):
        for right in range(left + 1, len(image_paths)):
            score = float(np.dot(embeddings[left], embeddings[right]))
            target = (
                same_scores
                if identity_key(image_paths[left]) == identity_key(image_paths[right])
                else different_scores
            )
            target.append(score)

    summary = {
        "images": len(image_paths),
        "embedding_dimension": int(embeddings[0].size),
        "all_finite": True,
        "same_pair_count": len(same_scores),
        "different_pair_count": len(different_scores),
        "same_median_cosine": float(np.median(same_scores)),
        "same_min_cosine": float(np.min(same_scores)),
        "different_median_cosine": float(np.median(different_scores)),
        "different_max_cosine": float(np.max(different_scores)),
    }
    summary["median_margin"] = (
        summary["same_median_cosine"] - summary["different_median_cosine"]
    )
    print(json.dumps(summary, indent=2))

    if summary["embedding_dimension"] != 512 or summary["median_margin"] <= 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
