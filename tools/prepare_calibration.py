#!/usr/bin/env python3
"""Build representative detector and recognizer calibration sets.

The source directory should contain authorized camera frames and/or public test
images.  The script uses the original SCRFD ONNX model to create aligned 112x112
face crops, then applies deterministic light/color/blur variations.  It never
uploads images.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


CANONICAL = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


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


def nms(boxes, scores, threshold=0.4):
    if not boxes:
        return []
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[current, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[current, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[current, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[current, 3], boxes[rest, 3])
        intersection = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_current = max(0.0, boxes[current, 2] - boxes[current, 0]) * max(
            0.0, boxes[current, 3] - boxes[current, 1]
        )
        area_rest = np.maximum(0.0, boxes[rest, 2] - boxes[rest, 0]) * np.maximum(
            0.0, boxes[rest, 3] - boxes[rest, 1]
        )
        iou = intersection / np.maximum(area_current + area_rest - intersection, 1e-6)
        order = rest[iou <= threshold]
    return keep


class ScrfdOnnx:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, image, threshold=0.5):
        canvas, scale, pad_x, pad_y = letterbox(image)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)
        tensor = ((rgb - 127.5) / 128.0).transpose(2, 0, 1)[None]
        outputs = self.session.run(None, {self.input_name: tensor})
        boxes, landmarks, scores = [], [], []
        for level, stride in enumerate((8, 16, 32)):
            score_data = np.asarray(outputs[level]).reshape(-1)
            box_data = np.asarray(outputs[level + 3]).reshape(-1, 4)
            point_data = np.asarray(outputs[level + 6]).reshape(-1, 10)
            feature_width = 640 // stride
            feature_height = 640 // stride
            spatial = feature_width * feature_height
            anchors = max(1, score_data.size // spatial)
            for index in np.flatnonzero(score_data >= threshold):
                location = int(index) // anchors
                x = location % feature_width
                y = location // feature_width
                center = np.array([x * stride, y * stride], dtype=np.float32)
                distances = box_data[index] * stride
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
                points = point_data[index].reshape(5, 2) * stride + center
                points[:, 0] = (points[:, 0] - pad_x) / scale
                points[:, 1] = (points[:, 1] - pad_y) / scale
                boxes.append(box)
                landmarks.append(points)
                scores.append(float(score_data[index]))

        kept = nms(boxes, scores)
        return [(boxes[i], landmarks[i], scores[i]) for i in kept]


def align_face(image, landmarks):
    transform, _ = cv2.estimateAffinePartial2D(
        np.asarray(landmarks, dtype=np.float32), CANONICAL, method=cv2.LMEDS
    )
    if transform is None:
        return None
    return cv2.warpAffine(image, transform, (112, 112))


def augment(image, index):
    alpha = (0.78, 0.9, 1.0, 1.1, 1.22)[index % 5]
    beta = (-18, -8, 0, 8, 18)[(index // 5) % 5]
    output = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    channel = index % 3
    if index % 4 == 1:
        shifted = output.astype(np.int16)
        shifted[:, :, channel] = np.clip(shifted[:, :, channel] + 10, 0, 255)
        output = shifted.astype(np.uint8)
    if index % 6 == 2:
        output = cv2.GaussianBlur(output, (3, 3), 0.7)
    if index % 7 == 3:
        output = cv2.flip(output, 1)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--det-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", type=int, default=12)
    args = parser.parse_args()

    image_paths = sorted(
        path
        for path in args.source.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not image_paths:
        parser.error(f"no source images found under {args.source}")
    if not args.det_model.is_file():
        parser.error(f"missing detector model: {args.det_model}")

    det_dir = args.output / "detect"
    rec_dir = args.output / "recognition"
    det_dir.mkdir(parents=True, exist_ok=True)
    rec_dir.mkdir(parents=True, exist_ok=True)
    detector = ScrfdOnnx(args.det_model)
    det_paths, rec_paths = [], []

    for source_index, image_path in enumerate(image_paths):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        faces = detector.detect(image)
        aligned_faces = []
        for _, landmarks, _ in faces:
            aligned = align_face(image, landmarks)
            if aligned is not None:
                aligned_faces.append(aligned)

        for variant in range(args.variants):
            det_image = augment(image, variant)
            det_path = det_dir / f"det_{source_index:03d}_{variant:02d}.jpg"
            cv2.imwrite(str(det_path), det_image, [cv2.IMWRITE_JPEG_QUALITY, 94])
            det_paths.append(det_path.resolve())

        for face_index, aligned in enumerate(aligned_faces):
            for variant in range(args.variants):
                rec_image = augment(aligned, variant)
                rec_path = rec_dir / (
                    f"rec_{source_index:03d}_{face_index:02d}_{variant:02d}.jpg"
                )
                cv2.imwrite(str(rec_path), rec_image, [cv2.IMWRITE_JPEG_QUALITY, 96])
                rec_paths.append(rec_path.resolve())

    if not rec_paths:
        raise SystemExit("No face was detected; recognition calibration set is empty")

    (args.output / "det_dataset.txt").write_text(
        "\n".join(str(path) for path in det_paths) + "\n", encoding="utf-8"
    )
    (args.output / "rec_dataset.txt").write_text(
        "\n".join(str(path) for path in rec_paths) + "\n", encoding="utf-8"
    )
    print(f"detector calibration images: {len(det_paths)}")
    print(f"recognizer calibration images: {len(rec_paths)}")


if __name__ == "__main__":
    main()

