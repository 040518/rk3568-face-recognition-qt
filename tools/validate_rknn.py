#!/usr/bin/env python3
"""Board-side FP16/INT8 regression for the two face models."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from rknnlite.api import RKNNLite


def load_model(path):
    runtime = RKNNLite()
    ret = runtime.load_rknn(str(path))
    if ret != 0:
        raise RuntimeError(f"load_rknn failed: {path}, ret={ret}")
    ret = runtime.init_runtime()
    if ret != 0:
        runtime.release()
        raise RuntimeError(f"init_runtime failed: {path}, ret={ret}")
    return runtime


def letterbox(image, size=640):
    scale = min(size / image.shape[1], size / image.shape[0])
    width = max(1, round(image.shape[1] * scale))
    height = max(1, round(image.shape[0] * scale))
    resized = cv2.resize(image, (width, height))
    left = (size - width) // 2
    top = (size - height) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top : top + height, left : left + width] = resized
    return canvas, float(scale), float(left), float(top)


def iou(a, b):
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-6)


def nms(faces, threshold=0.4):
    faces = sorted(faces, key=lambda face: face[2], reverse=True)
    kept = []
    for face in faces:
        if all(iou(face[0], other[0]) <= threshold for other in kept):
            kept.append(face)
    return kept


def decode_scrfd(outputs, scale, pad_x, pad_y, threshold=0.5):
    faces = []
    for level, stride in enumerate((8, 16, 32)):
        scores = np.asarray(outputs[level], dtype=np.float32).reshape(-1)
        boxes = np.asarray(outputs[level + 3], dtype=np.float32).reshape(-1, 4)
        points = np.asarray(outputs[level + 6], dtype=np.float32).reshape(-1, 10)
        feature_width = 640 // stride
        feature_height = 640 // stride
        spatial = feature_width * feature_height
        anchors = max(1, scores.size // spatial)
        for index in np.flatnonzero(scores >= threshold):
            location = int(index) // anchors
            x = location % feature_width
            y = location // feature_width
            center = np.array([x * stride, y * stride], dtype=np.float32)
            distances = boxes[index] * stride
            box = np.array(
                [
                    center[0] - distances[0],
                    center[1] - distances[1],
                    center[0] + distances[2],
                    center[1] + distances[3],
                ],
                dtype=np.float32,
            )
            box[[0, 2]] = (box[[0, 2]] - pad_x) / scale
            box[[1, 3]] = (box[[1, 3]] - pad_y) / scale
            landmarks = points[index].reshape(5, 2) * stride + center
            landmarks[:, 0] = (landmarks[:, 0] - pad_x) / scale
            landmarks[:, 1] = (landmarks[:, 1] - pad_y) / scale
            faces.append((box, landmarks, float(scores[index])))
    return nms(faces)


def run_detector(model_path, image_paths):
    runtime = load_model(model_path)
    results = []
    try:
        for image_path in image_paths:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"cannot read image: {image_path}")
            canvas, scale, pad_x, pad_y = letterbox(image)
            rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            outputs = runtime.inference(inputs=[rgb])
            results.append(decode_scrfd(outputs, scale, pad_x, pad_y))
    finally:
        runtime.release()
    return results


def run_recognizer(model_path, image_paths):
    runtime = load_model(model_path)
    embeddings = []
    try:
        for image_path in image_paths:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"cannot read image: {image_path}")
            image = cv2.resize(image, (112, 112))
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            output = np.asarray(runtime.inference(inputs=[rgb])[0], dtype=np.float32)
            embedding = output.reshape(-1)
            embedding /= max(float(np.linalg.norm(embedding)), 1e-12)
            embeddings.append(embedding)
    finally:
        runtime.release()
    return embeddings


def identity_key(path):
    parts = path.stem.split("_")
    return "_".join(parts[:3])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--det-fp", type=Path, required=True)
    parser.add_argument("--det-int8", type=Path, required=True)
    parser.add_argument("--rec-fp", type=Path, required=True)
    parser.add_argument("--rec-int8", type=Path, required=True)
    parser.add_argument("--det-images", type=Path, required=True)
    parser.add_argument("--rec-images", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.45)
    args = parser.parse_args()

    det_images = sorted(args.det_images.glob("*.jpg"))[:12]
    rec_images = sorted(args.rec_images.glob("*.jpg"))[:64]
    if not det_images or not rec_images:
        parser.error("validation image directories are empty")

    det_fp = run_detector(args.det_fp, det_images)
    det_int8 = run_detector(args.det_int8, det_images)
    box_ious, landmark_errors = [], []
    count_matches = 0
    for image_path, fp_faces, int_faces in zip(det_images, det_fp, det_int8):
        if len(fp_faces) == len(int_faces):
            count_matches += 1
        used = set()
        for fp_face in fp_faces:
            candidates = [
                (iou(fp_face[0], int_face[0]), index, int_face)
                for index, int_face in enumerate(int_faces)
                if index not in used
            ]
            if not candidates:
                continue
            overlap, index, int_face = max(candidates, key=lambda item: item[0])
            used.add(index)
            box_ious.append(overlap)
            diagonal = max(
                float(np.hypot(fp_face[0][2] - fp_face[0][0],
                               fp_face[0][3] - fp_face[0][1])),
                1.0,
            )
            error = np.linalg.norm(fp_face[1] - int_face[1], axis=1).mean() / diagonal
            landmark_errors.append(float(error))
        print(f"DET {image_path.name} fp={len(fp_faces)} int8={len(int_faces)}")

    rec_fp = run_recognizer(args.rec_fp, rec_images)
    rec_int8 = run_recognizer(args.rec_int8, rec_images)
    embedding_cosines = [
        float(np.dot(fp, quantized))
        for fp, quantized in zip(rec_fp, rec_int8)
    ]

    decision_flips = 0
    pair_count = 0
    for left in range(len(rec_images)):
        for right in range(left + 1, len(rec_images)):
            same_identity = identity_key(rec_images[left]) == identity_key(rec_images[right])
            fp_score = float(np.dot(rec_fp[left], rec_fp[right]))
            int8_score = float(np.dot(rec_int8[left], rec_int8[right]))
            fp_decision = fp_score >= args.threshold
            int8_decision = int8_score >= args.threshold
            if fp_decision != int8_decision:
                decision_flips += 1
            pair_count += 1
            if same_identity and (not fp_decision or not int8_decision):
                print(
                    f"WARN same-person below threshold: "
                    f"{rec_images[left].name} {rec_images[right].name} "
                    f"fp={fp_score:.4f} int8={int8_score:.4f}"
                )

    summary = {
        "det_images": len(det_images),
        "det_count_matches": count_matches,
        "det_median_iou": float(np.median(box_ious)) if box_ious else 0.0,
        "det_mean_landmark_error": float(np.mean(landmark_errors))
        if landmark_errors
        else 1.0,
        "rec_images": len(rec_images),
        "rec_min_fp_int8_cosine": min(embedding_cosines),
        "rec_median_fp_int8_cosine": float(np.median(embedding_cosines)),
        "decision_pairs": pair_count,
        "decision_flips": decision_flips,
    }
    summary["passed"] = (
        count_matches == len(det_images)
        and summary["det_median_iou"] >= 0.90
        and summary["det_mean_landmark_error"] <= 0.015
        and summary["rec_min_fp_int8_cosine"] >= 0.98
        and decision_flips == 0
    )
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False, sort_keys=True))
    raise SystemExit(0 if summary["passed"] else 3)


if __name__ == "__main__":
    main()

