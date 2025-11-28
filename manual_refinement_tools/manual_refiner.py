import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk, ImageDraw
import json
import os
import shutil
import numpy as np
from pathlib import Path
import sys
import argparse


class MallTrackingRefiner:
    def __init__(self, root, input_dir, output_dir):
        self.root = root
        self.root.title("Mall Tracking Refiner (Human-in-the-Loop)")
        self.root.geometry("1400x900")

        self.json_dir = input_dir / "json"
        self.img_dir = input_dir / "images"

        self.out_json = output_dir / "json"
        self.out_img = output_dir / "images"

        self.out_json.mkdir(parents=True, exist_ok=True)
        self.out_img.mkdir(parents=True, exist_ok=True)

        self.files = sorted(list(self.json_dir.glob("*.json")))

        if not self.files:
            messagebox.showerror(
                "Error", f"No 'meta_*.json' files found in:\n{self.json_dir}"
            )
            sys.exit(1)

        self.current_idx = 0
        self.current_data = []
        self.selected_obj_idx = -1
        self.scale_factor = 1.0

        self.setup_ui()
        self.load_frame()

        self.root.bind("<Right>", lambda e: self.next_frame())
        self.root.bind("<Left>", lambda e: self.prev_frame())
        self.root.bind("<Delete>", lambda e: self.delete_object())
        self.root.bind("m", lambda e: self.toggle_status())

    def setup_ui(self):
        # main canvas (left)
        self.canvas_frame = tk.Frame(self.root, bg="#333333")
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # control panel (right)
        self.controls = tk.Frame(
            self.root, width=320, bg="#f0f0f0", relief=tk.RAISED, borderwidth=1
        )
        self.controls.pack(side=tk.RIGHT, fill=tk.Y)
        self.controls.pack_propagate(False)

        # header
        tk.Label(
            self.controls,
            text="Refiner Controls",
            font=("Segoe UI", 16, "bold"),
            bg="#f0f0f0",
        ).pack(pady=15)
        self.lbl_frame = tk.Label(
            self.controls, text="Frame: 0/0", font=("Consolas", 10), bg="#f0f0f0"
        )
        self.lbl_frame.pack(pady=5)

        # edit section
        self.frame_edit = tk.LabelFrame(
            self.controls,
            text="Selection Details",
            padx=10,
            pady=10,
            font=("Segoe UI", 11, "bold"),
            bg="#f0f0f0",
        )
        self.frame_edit.pack(fill=tk.X, padx=10, pady=10)

        # Global ID Input
        tk.Label(self.frame_edit, text="Global ID:", bg="#f0f0f0").grid(
            row=0, column=0, sticky="w"
        )
        self.ent_id = tk.Entry(self.frame_edit, width=8, font=("Consolas", 12))
        self.ent_id.grid(row=0, column=1, padx=5)
        tk.Button(
            self.frame_edit, text="Update", command=self.update_id, bg="#e1e1e1"
        ).grid(row=0, column=2)

        # Status Toggle
        tk.Label(self.frame_edit, text="Status:", bg="#f0f0f0").grid(
            row=1, column=0, sticky="w", pady=10
        )
        self.lbl_status = tk.Label(
            self.frame_edit,
            text="-",
            font=("Segoe UI", 10, "bold"),
            bg="#f0f0f0",
            width=8,
        )
        self.lbl_status.grid(row=1, column=1)
        tk.Button(
            self.frame_edit, text="Flip (M)", command=self.toggle_status, bg="#e1e1e1"
        ).grid(row=1, column=2)

        # Delete Button
        tk.Button(
            self.frame_edit,
            text="DELETE OBJECT (Del)",
            bg="#ffcccc",
            fg="red",
            font=("Segoe UI", 10, "bold"),
            command=self.delete_object,
        ).grid(row=2, column=0, columnspan=3, pady=15, sticky="ew")

        # Navigation Footer
        self.frame_nav = tk.Frame(self.controls, bg="#f0f0f0")
        self.frame_nav.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=20)

        tk.Button(
            self.frame_nav, text="< Prev", command=self.prev_frame, width=10, height=2
        ).pack(side=tk.LEFT)
        tk.Button(
            self.frame_nav,
            text="Save & Next >",
            command=self.next_frame,
            bg="#ccffcc",
            width=15,
            height=2,
        ).pack(side=tk.RIGHT)

    def load_frame(self):
        json_path = self.files[self.current_idx]
        img_filename = f"{json_path.stem[1:]}.jpg"
        img_path = self.img_dir / img_filename

        if not img_path.exists():
            messagebox.showerror("Error", f"Image file not found:\n{img_path}")
            return

        with open(json_path, "r") as f:
            self.current_data = json.load(f)

        pil_img = Image.open(img_path)

        # to fit into the window
        screen_w = self.root.winfo_width() - 320  # Subtract control panel
        screen_h = self.root.winfo_height()

        if screen_w < 100:
            screen_w = 1000  # Fallback during init
        if screen_h < 100:
            screen_h = 800

        w_ratio = screen_w / pil_img.width
        h_ratio = screen_h / pil_img.height
        self.scale_factor = min(w_ratio, h_ratio, 1.0)

        new_w = int(pil_img.width * self.scale_factor)
        new_h = int(pil_img.height * self.scale_factor)

        self.display_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(self.display_img)

        # reset selection
        self.selected_obj_idx = -1
        self.lbl_frameconfig(
            text=f"{json_path.name} ({self.current_idx + 1}/{len(self.files)})"
        )
        self.redraw()

    def redraw(self):
        self.canvas.delete("all")

        # might have ankor bugs need to check
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

        for idx, obj in enumerate(self.current_data):
            x1, y1, x2, y2 = [c * self.scale_factor for c in obj["bbox"]]
            gid = obj.get("global_id", -1)
            status = obj.get("status", "Unknown")

            if idx == self.selected_obj_idx:
                outline = "yellow"
                width = 3
                fill_txt = "yellow"
            elif status == "Moving":
                outline = "#ff3333"  # Red
                width = 2
                fill_txt = "#ff3333"
            else:
                outline = "#3399ff"  # Blue
                width = 2
                fill_txt = "#3399ff"

            # Draw Box
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=outline, width=width)

            # Draw Label
            label_text = f"{gid}"
            self.canvas.create_text(
                x1,
                y1 - 10,
                text=label_text,
                fill=fill_txt,
                anchor=tk.SW,
                font=("Arial", 12, "bold"),
            )

            # Draw polygon outline (if available)
            if "segmentation_polygon" in obj and obj["segmentation_polygon"]:
                for poly in obj["segmentation_polygon"]:
                    scaled_poly = [c * self.scale_factor for c in poly]
                    self.canvas.create_polygon(
                        scaled_poly, outline=outline, fill="", width=1, dash=(2, 2)
                    )

        self.update_control_panel()

    def on_canvas_click(self, event):
        click_x, click_y = event.x, event.y
        best_idx = -1
        min_area = float("inf")

        # Hit testing (Box)
        for idx, obj in enumerate(self.current_data):
            x1, y1, x2, y2 = [c * self.scale_factor for c in obj["bbox"]]
            if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                # select smalles overlapping box
                area = (x2 - x1) * (y2 - y1)
                if area < min_area:
                    min_area = area
                    best_idx = idx

        self.selected_obj_idx = best_idx
        self.redraw()

    def update_control_panel(self):
        if self.selected_obj_idx >= 0:
            obj = self.current_data[self.selected_obj_idx]
            self.ent_id.delete(0, tk.END)
            self.ent_id.insert(0, str(obj.get("global_id", "")))
            self.lbl_status.config(text=obj.get("status", "Unknown"))
            self.frame_edit.config(
                text=f"Editing Object #{self.selected_obj_idx}", fg="black"
            )
        else:
            self.ent_id.delete(0, tk.END)
            self.lbl_status.config(text="-")
            self.frame_edit.config(text="No Selection", fg="gray")

    def update_id(self):
        if self.selected_obj_idx >= 0:
            try:
                new_id = int(self.ent_id.get())
                self.current_data[self.selected_obj_idx]["global_id"] = new_id
                self.redraw()
            except ValueError:
                messagebox.showerror("Input Error", "ID must be a number.")

    def toggle_status(self):
        if self.selected_obj_idx >= 0:
            curr = self.current_data[self.selected_obj_idx].get("status", "Static")
            new_status = "Static" if curr == "Moving" else "Moving"
            self.current_data[self.selected_obj_idx]["status"] = new_status
            self.redraw()

    def delete_object(self):
        if self.selected_obj_idx >= 0:
            del self.current_data[self.selected_obj_idx]
            self.selected_obj_idx = -1
            self.redraw()

    def next_frame(self):
        self.save_current()
        if self.current_idx < len(self.files) - 1:
            self.current_idx += 1
            self.load_frame()
        else:
            messagebox.showinfo("Finished", "You have reached the end of the sequence!")

    def prev_frame(self):
        self.save_current()
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_frame()

    def save_current(self):
        src_json = self.files[self.current_idx]
        dst_json = self.out_json / src_json.name
        with open(dst_json, "w") as f:
            json.dump(self.current_data, f, indent=2)

        # Copy Original Image (for completeness)
        img_filename = f"{src_json.stem[1]}.jpg"
        src_img = self.img_dir / img_filename
        dst_img = self.out_img / img_filename
        if src_img.exists() and not dst_img.exists():
            shutil.copy(src_img, dst_img)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Manual Refiner Tool for Mall Tracking Data"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--cam", type=str, help="Camera ID to edit (e.g., '1' or 'cam_1')"
    )
    group.add_argument(
        "--input_dir", type=str, help="Full path to a specific 'cam_X' folder"
    )
    parser.add_argument(
        "--root",
        type=str,
        default="./output_data",
        help="Root output directory (default: ./output_data)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.input_dir:
        # Use full path provided
        input_path = Path(args.input_dir)
    else:
        # Construct path from Camera ID
        cam_name = args.cam if args.cam.startswith("cam_") else f"cam_{args.cam}"
        input_path = Path(args.root) / cam_name

    # Validate Input Path
    if not input_path.exists():
        print(f"\n[ERROR] Directory not found: {input_path}")
        print(f"Please check your --root or --cam arguments.\n")
        sys.exit(1)

    # Determine Output Path (Auto-create '_corrected')
    output_path = input_path.parent / f"{input_path.name}_corrected"

    print(f"\n--- LAUNCHING REFINER ---")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print("-" * 30)

    # Launch GUI
    root = tk.Tk()
    # Center window fix
    root.eval("tk::PlaceWindow . center")
    app = MallTrackingRefiner(root, input_path, output_path)
    root.mainloop()
