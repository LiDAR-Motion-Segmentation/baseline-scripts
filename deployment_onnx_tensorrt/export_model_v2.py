import argparse
import torch
from ultralytics import YOLO, SAM
from pathlib import Path


def export_model(model_type: str, weights_path: str, export_format: str):
    """
    Exports a given model to ONNX or TensorRT format.

    Args:
        model_type (str): The type of model to export ('yolo' or 'sam').
        weights_path (str): Path to the model's .pt weights file.
        export_format (str): The format to export to ('onnx' or 'tensorrt').
    """

    print(f"Loading {model_type} model from: {weights_path}")
    if model_type == "yolo":
        model = YOLO(weights_path)
    elif model_type == "sam":
        # IMPORTANT: This MUST be an Ultralytics-compatible SAM model (e.g., sam_l.pt)
        model = SAM(weights_path)
    else:
        raise ValueError("Invalid model_type. Choose 'yolo' or 'sam'.")

    output_path = Path(weights_path)
    output_name = f"{output_path.stem}.{export_format}"

    export_args = {
        "format": export_format,
        "half": True,  # Use FP16 precision for faster inference
        "opset": 12,  # A good default for ONNX compatibility
    }

    if export_format == "tensorrt":
        # TensorRT requires a specific workspace size (in GB)
        export_args["workspace"] = 8
        print("TensorRT export will take several minutes...")

    if model_type == "yolo":
        # For YOLO, it's good to specify dynamic batch/dimensions for flexibility
        export_args["dynamic"] = True
        export_args["imgsz"] = 640

    if model_type == "sam":
        # SAM's encoder is static, specify image size
        export_args["imgsz"] = 1024
        print("Note: Exporting SAM will create two models: an encoder and a decoder.")

    # --- 3. Run Export ---
    print(f"Exporting {model_type} to {export_format}...")
    try:
        exported_path = model.export(**export_args)
        print(f"Successfully exported to: {exported_path}")
    except Exception as e:
        print(f"\n--- EXPORT FAILED ---")
        print(f"Error: {e}")
        if "sam_hq_vit" in weights_path:
            print(
                "CRITICAL_ERROR: You are trying to export an incompatible HQ-SAM model."
            )
            print("Please download an Ultralytics-compatible model like 'sam_l.pt'.")
        if export_format == "tensorrt":
            print(
                "TensorRT export can fail if 'polygraphy' or 'onnx-graphsurgeon' are missing."
            )
            print("Try running: pip install onnx-graphsurgeon polygraphy")


def main():
    parser = argparse.ArgumentParser(description="Model Exporter for YOLO and SAM")
    parser.add_argument(
        "--weights", type=str, required=True, help="Path to the .pt model weights file."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["yolo", "sam"],
        help="Type of model to export: 'yolo' or 'sam'.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="onnx",
        choices=["onnx", "tensorrt"],
        help="Format to export to: 'onnx' (recommended first) or 'tensorrt' (for NVIDIA GPU).",
    )
    args = parser.parse_args()

    export_model(args.model, args.weights, args.format)


if __name__ == "__main__":
    main()
