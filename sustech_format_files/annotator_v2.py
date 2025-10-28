import cv2
import numpy as np
import random
from pathlib import Path
from ultralytics import YOLO, SAM
import os


def count_files(dir: str) -> int:
    count = 0
    for i in os.scandir(dir):
        if i.is_file():
            count += 1
    return count


def auto_annotate(
    data: str | Path,
    det_model: str = "yolov8x.pt",
    sam_model: str = "sam_b.pt",
    device: str = "",
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    max_det: int = 300,
    classes: list[int] | None = None,
    output_dir: str | Path | None = None,
    visualize: bool = False,
) -> None:

    det_model = YOLO(det_model)
    sam_model = SAM(sam_model)

    data = Path(data)
    if not output_dir:
        output_dir = data.parent / f"{data.stem}_auto_annotated"

    box_dir = Path(output_dir) / "box_labels"
    seg_dir = Path(output_dir) / "segment_labels"
    viz_dir = Path(output_dir) / "visualizations"

    box_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)

    if visualize:
        viz_dir.mkdir(parents=True, exist_ok=True)
        colors = [
            [random.randint(0, 255) for _ in range(3)]
            for _ in range(len(det_model.names))
        ]

    det_results = det_model(
        data,
        stream=True,
        device=device,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        max_det=max_det,
        classes=classes,
    )

    for result in det_results:
        if class_ids := result.boxes.cls.int().tolist():
            boxes_xywhn = result.boxes.xywhn.cpu().numpy()
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()

            box_label_path = box_dir / f"{Path(result.path).stem}.txt"
            with open(box_label_path, "w", encoding="utf-8") as f_box:
                for i, box in enumerate(boxes_xywhn):
                    x_center, y_center, width, height = box
                    f_box.write(
                        f"{class_ids[i]} {x_center:.5f} {y_center:.5f} {width:.5f} {height:.5f}\n"
                    )

            sam_results = sam_model(
                result.orig_img,
                bboxes=boxes_xyxy,
                verbose=False,
                save=False,
                device=device,
            )
            segments = sam_results[0].masks.xyn
            seg_label_path = seg_dir / f"{Path(result.path).stem}.txt"
            with open(seg_label_path, "w", encoding="utf-8") as f_seg:
                for i, s in enumerate(segments):
                    if s.any():
                        segment_str = " ".join(
                            map(lambda x: f"{x:.5f}", s.reshape(-1).tolist())
                        )
                        f_seg.write(f"{class_ids[i]} {segment_str}\n")

            if visualize:
                annotated_img = result.orig_img.copy()
                H, W, _ = annotated_img.shape
                overlay = annotated_img.copy()
                alpha = 0.5
                segmentation_mask = np.zeros((H, W), dtype=np.uint8)

                timestep = count_files(
                    dir="/media/soumoroy/Extreme SSD/Motion-segementation-rosbags/tech_fest/data/husky_dataset/equirectangular_auto_annotate_labels/"
                )
                timestep_p = timestep + 3
                filepath = f"/media/soumoroy/Extreme SSD/Motion-segementation-rosbags/tech_fest/data/husky_dataset/equirectangular_auto_annotate_labels/{timestep_p:06d}.txt"
                all_polygons = []
                with open(filepath, "r") as f:
                    for line in f:
                        poly_data = np.array([float(x) for x in line.split()])
                        all_polygons.append(poly_data)

                for poly_data in all_polygons:
                    normalized_points = poly_data[1:].reshape(-1, 2)

                for i, seg_points_normalized in enumerate(segments):
                    if not seg_points_normalized.any():
                        continue
                    color = colors[class_ids[i]]

                    normalized_points = poly_data[1:].reshape(-1, 2)
                    pixel_points = (normalized_points * np.array([W, H])).astype(
                        np.int32
                    )

                    seg_points = (seg_points_normalized * np.array([W, H])).astype(
                        np.int32
                    )
                    cv2.fillPoly(overlay, [seg_points], color)
                    cv2.fillPoly(segmentation_mask, [pixel_points], color=1)

                # blending the overlay on the orginal image
                cv2.addWeighted(
                    overlay, alpha, annotated_img, 1 - alpha, 0, annotated_img
                )

                for i, box in enumerate(boxes_xyxy):
                    x1, y1, x2, y2 = map(int, box)
                    class_id = class_ids[i]
                    color = colors[class_id]
                    label = f"{det_model.names[class_id]}"
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        annotated_img,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                    )

                viz_path = viz_dir / Path(result.path).name
                cv2.imwrite(str(viz_path), annotated_img)
