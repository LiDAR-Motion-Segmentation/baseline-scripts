#!/usr/bin/env python3

import os
import argparse
import subprocess
import torch
from ultralytics import YOLO
from segment_anything import sam_model_registery
from segment_anything.onnx_scripts.export_onnx_model import sam_model_to_onnx


def export_yolov8_to_onnx(yolo_model_path, out_path):
    model = YOLO(yolo_model_path)
    model.export(
        format="onnx",
        export_dir=os.path.dirname(out_path),
        dynamic=True,
        opset=12,
        simplify=True,
    )
    print(f"YOLOv8 model exported to {out_path}")


def export_yolov8_to_trt(yolo_model_path, onnx_out, engine_out):
    export_yolov8_to_onnx(yolo_model_path, onnx_out)
    print("Building TensorRT engine (needs trtexec, may require sudo)...")
    cmd = ["trtexec", f"--onnx={onnx_out}", f"--saveEngine={engine_out}", "--fp16"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"TensorRT engine saved to {engine_out}")
    else:
        print("TensorRT conversion failed! Output:")
        print(result.stdout)
        print(result.stderr)


def export_sam_hq_to_onnx(sam_ckpt_path, out_path):
    pass


def export_sam_hq_to_trt(sam_ckpt_path, onnx_out, engine_out):
    print("Building TensorRT engine for SAM (needs trtexec)...")
    cmd = ["trtexec", f"--onnx={onnx_out}", f"--saveEngine={engine_out}", "--fp16"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"TensorRT engine saved to {engine_out}")
    else:
        print("TensorRT conversion failed! Output:")
        print(result.stdout)
        print(result.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Export YOLOv8 and SAM-HQ to ONNX or TensorRT."
    )
    parser.add_argument("--yolo", required=True, help="Path to YOLOv8 .pt model")
    parser.add_argument(
        "--sam", required=True, help="Path to sam_hq_vit_h.pth checkpoint"
    )
    parser.add_argument(
        "--mode",
        choices=["onnx", "tensorrt"],
        default="onnx",
        help='Export to "onnx" or "tensorrt" (.engine via .onnx)',
    )
    parser.add_argument(
        "--output_dir",
        default="./exported",
        help="Directory where exported models are saved",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    yolo_onnx = os.path.join(args.output_dir, "yolov8l.onnx")
    yolo_engine = os.path.join(args.output_dir, "yolov8l.engine")
    sam_onnx = os.path.join(args.output_dir, "sam_hq_vit_h.onnx")
    sam_engine = os.path.join(args.output_dir, "sam_hq_vit_h.engine")

    if args.mode == "onnx":
        export_yolov8_to_onnx(args.yolo, yolo_onnx)
        export_sam_hq_to_onnx(args.sam, sam_onnx)
    elif args.mode == "tensorrt":
        export_yolov8_to_trt(args.yolo, yolo_onnx, yolo_engine)
        export_sam_hq_to_trt(args.sam, sam_onnx, sam_engine)


if __name__ == "__main__":
    main()
