import os
import sam3
import torch

# Idk why but without a forceful path hardcoding stuff doesnt work
sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")

from sam3.model_builder import build_sam3_video_predictor

import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sam3.visualization_utils import (
    load_frame,
    prepare_masks_for_visualization,
    visualize_formatted_frame_output,
    masks_to_boxes,
)

# (This handles the dependencies like plot_bbox, COLORS, etc.)
import sam3.visualization_utils as vis_utils

plot_bbox = getattr(vis_utils, "plot_bbox", None)
normalize_bbox = getattr(vis_utils, "normalize_bbox", None)
plot_mask = getattr(vis_utils, "plot_mask", None)
show_points = getattr(vis_utils, "show_points", None)
COLORS = getattr(
    vis_utils, "COLORS", ["#00FF00", "#FF0000", "#0000FF"]
)  # Fallback colors

# use all available GPUs on the machine
# as of now rtx A4000
gpus_to_use = [0]
predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)

# "video_path" needs to be either a JPEG folder or a MP4 video file
# hardcoding as of now will make an argument later to feed it in a modular manner
video_path = "/scratch/soumo_roy/sam3/examples/output_short.mp4"
# Run your loop
save_dir = "/scratch/soumo_roy/sam3/examples/debug_frames"

# font size for axes titles
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["figure.titlesize"] = 12


def propagate_in_video(predictor, session_id):
    # we will just propagate from frame 0 to the end of the video
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]

    return outputs_per_frame


def abs_to_rel_coords(coords, IMG_WIDTH, IMG_HEIGHT, coord_type="point"):
    """Convert absolute coordinates to relative coordinates (0-1 range)

    Args:
        coords: List of coordinates
        coord_type: 'point' for [x, y] or 'box' for [x, y, w, h]
    """
    if coord_type == "point":
        return [[x / IMG_WIDTH, y / IMG_HEIGHT] for x, y in coords]
    elif coord_type == "box":
        return [
            [x / IMG_WIDTH, y / IMG_HEIGHT, w / IMG_WIDTH, h / IMG_HEIGHT]
            for x, y, w, h in coords
        ]
    else:
        raise ValueError(f"Unknown coord_type: {coord_type}")


# need to put this below in a function I guess
# load "video_frames_for_vis" for visualization purposes (they are not used by the model)
if isinstance(video_path, str) and video_path.endswith(".mp4"):
    cap = cv2.VideoCapture(video_path)
    video_frames_for_vis = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        video_frames_for_vis.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
else:
    video_frames_for_vis = glob.glob(os.path.join(video_path, "*.png"))
    try:
        # integer sort instead of string sort (so that e.g. "2.jpg" is before "11.jpg")
        video_frames_for_vis.sort(
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
        )
    except ValueError:
        # fallback to lexicographic sort if the format is not "<frame_index>.jpg"
        print(
            f'frame names are not in "<frame_index>.jpg" format: {video_frames_for_vis[:5]=}, '
            f"falling back to lexicographic sort."
        )
        video_frames_for_vis.sort()

# prolly need an inference function
response = predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=video_path,
    )
)
session_id = response["session_id"]

# it's required to reset the session first (otherwise the results would be wrong)
_ = predictor.handle_request(
    request=dict(
        type="reset_session",
        session_id=session_id,
    )
)

# should take prompt as an argument
prompt_text_str = "person"
frame_idx = 0  # add a text prompt on frame 0
response = predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=frame_idx,
        text=prompt_text_str,
    )
)
out = response["outputs"]

# I dont think we need this below
plt.close("all")
visualize_formatted_frame_output(
    frame_idx,
    video_frames_for_vis,
    outputs_list=[prepare_masks_for_visualization({frame_idx: out})],
    titles=["SAM 3 Dense Tracking outputs"],
    figsize=(6, 4),
)

# now we propagate the outputs from frame 0 to the end of the video and collect all outputs
outputs_per_frame = propagate_in_video(predictor, session_id)

# finally, we reformat the outputs for visualization and plot the outputs every frames
outputs_per_frame = prepare_masks_for_visualization(outputs_per_frame)


# 2. Define the UPDATED function directly here
def visualize_formatted_frame_output(
    frame_idx,
    video_frames,
    outputs_list,
    titles=None,
    points_list=None,
    points_labels_list=None,
    figsize=(12, 8),
    title_suffix="",
    prompt_info=None,
    save_output=False,  # <--- ADDED
    output_dir="outputs",  # <--- ADDED
):
    """Visualize and optionally save segmentation masks."""

    # Handle single output dict case
    if isinstance(outputs_list, dict) and frame_idx in outputs_list:
        outputs_list = [outputs_list]
    elif isinstance(outputs_list, dict) and not any(
        isinstance(k, int) for k in outputs_list.keys()
    ):
        single_frame_outputs = {frame_idx: outputs_list}
        outputs_list = [single_frame_outputs]

    num_outputs = len(outputs_list)
    if titles is None:
        titles = [f"Set {i + 1}" for i in range(num_outputs)]

    # Create the plot
    fig, axes = plt.subplots(1, num_outputs, figsize=figsize)
    if num_outputs == 1:
        axes = [axes]

    img = load_frame(video_frames[frame_idx])
    img_H, img_W, _ = img.shape

    for idx in range(num_outputs):
        ax, outputs_set, ax_title = axes[idx], outputs_list[idx], titles[idx]
        ax.set_title(f"Frame {frame_idx} - {ax_title}{title_suffix}")
        ax.imshow(img)

        if frame_idx in outputs_set:
            _outputs = outputs_set[frame_idx]

            # Draw Objects
            objects_drawn = 0
            for obj_id, binary_mask in _outputs.items():
                if isinstance(binary_mask, torch.Tensor):
                    binary_mask = binary_mask.cpu()

                mask_sum = (
                    binary_mask.sum()
                    if hasattr(binary_mask, "sum")
                    else np.sum(binary_mask)
                )

                if mask_sum > 0:
                    if not isinstance(binary_mask, torch.Tensor):
                        binary_mask = torch.tensor(binary_mask)

                    # Get BBox
                    if binary_mask.any():
                        box_xyxy = masks_to_boxes(binary_mask.unsqueeze(0)).squeeze()
                        box_xyxy = normalize_bbox(box_xyxy, img_W, img_H)
                    else:
                        box_xyxy = [0.45, 0.45, 0.55, 0.55]

                    # Pick color
                    color = COLORS[obj_id % len(COLORS)]

                    if plot_bbox:
                        plot_bbox(
                            img_H,
                            img_W,
                            box_xyxy,
                            text=f"(id={obj_id})",
                            box_format="XYXY",
                            color=color,
                            ax=ax,
                        )

                    mask_np = binary_mask.numpy()
                    if plot_mask:
                        plot_mask(mask_np, color=color, ax=ax)
                    objects_drawn += 1

            if objects_drawn == 0:
                ax.text(
                    0.5,
                    0.5,
                    "No objects",
                    transform=ax.transAxes,
                    fontsize=16,
                    ha="center",
                    color="red",
                )

        ax.axis("off")

    plt.tight_layout()

    # --- SAVE LOGIC (Must happen BEFORE plt.show) ---
    if save_output:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{frame_idx:05d}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")

    # plt.show() # Commented out to prevent display spam and clearing memory
    plt.close(fig)  # Important: Free memory


print(f"Saving frames to {save_dir}...")

for frame_idx in range(len(outputs_per_frame)):
    visualize_formatted_frame_output(
        frame_idx,
        video_frames_for_vis,
        outputs_list=[outputs_per_frame],
        titles=["SAM 3 Dense Tracking outputs"],
        figsize=(6, 4),
        save_output=True,
        output_dir=save_dir,
    )
    if frame_idx % 10 == 0:
        print(f"Saved {frame_idx}/{len(outputs_per_frame)}", end="\r")

print("\nDone! All frames saved.")


## prolly need a main function to bind things together letsee
def main():
    pass


if "__name__" == "__main__":
    main()
