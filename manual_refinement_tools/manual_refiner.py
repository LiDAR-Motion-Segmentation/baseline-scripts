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
    def __init__(self, root, json_dir, img_dir, out_dir):
        self.root = root
        self.root.title(f"Refiner Tool : {json_dir.name} <-> {img_dir.name}")
        self.root.geometry("1650x850")

        self.json_dir = json_dir
        self.img_dir = img_dir

        self.out_json = out_dir / "json"
        self.out_img = out_dir / "images"
        self.out_labels = out_dir / "labels"

        self.out_json.mkdir(parents=True, exist_ok=True)
        self.out_img.mkdir(parents=True, exist_ok=True)
        self.out_labels.mkdir(parents=True, exist_ok=True)

        all_jsons = sorted(list(self.json_dir.glob("*.json")))

        self.files = [f for f in all_jsons if f.stem.isdigit()]

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
        input_json_path = self.files[self.current_idx]
        stem = input_json_path.stem  # e.g. "000123"

        # This ensures we load your previous edits
        output_json_path = self.out_json / input_json_path.name

        if output_json_path.exists():
            # Load the file you edited
            load_path = output_json_path
            print(f"Loading corrected: {output_json_path.name}")
        else:
            # First time loading: Load original
            load_path = input_json_path
            print(f"Loading original: {input_json_path.name}")

        with open(load_path, "r") as f:
            self.current_data = json.load(f)

        # SEARCH LOGIC: Check valid extensions
        valid_exts = [".png", ".jpg", ".jpeg", ".bmp"]
        found_img_path = None

        for ext in valid_exts:
            candidate = self.img_dir / f"{stem}{ext}"
            if candidate.exists():
                found_img_path = candidate
                break

        if found_img_path is None:
            messagebox.showerror(
                "Error", f"Image not found for {stem}.\nChecked: {valid_exts}"
            )
            return

        # Store the actual path for saving later
        self.current_img_path = found_img_path
        self.pil_img = Image.open(self.current_img_path)
        self.scale_factor = 1.0
        # # If you strictly want to force 1280x720 even if input is huge/small:
        # self.pil_img = self.pil_img.resize((1280, 720), Image.Resampling.LANCZOS)

        self.display_img = self.pil_img
        self.tk_img = ImageTk.PhotoImage(self.display_img)

        self.selected_obj_idx = -1
        self.lbl_frame.config(
            text=f"{input_json_path.name} ({self.current_idx + 1}/{len(self.files)})"
        )
        self.redraw()

    def redraw(self):
        self.canvas.delete("all")
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

        if hasattr(self, "current_img_path") and self.current_img_path.exists():
            dst_img = self.out_img / self.current_img_path.name
            if not dst_img.exists():
                shutil.copy(self.current_img_path, dst_img)

        # Export YOLO Labels
        if hasattr(self, "pil_img"):
            txt_filename = f"{src_json.stem}.txt"
            dst_txt = self.out_labels / txt_filename
            self.export_yolo_txt(dst_txt, self.pil_img.width, self.pil_img.height)

    def export_yolo_txt(self, output_path, img_w, img_h):
        with open(output_path, "w") as f:
            for obj in self.current_data:
                # class ID 0 for person
                class_id = 0

                if "segmentation_polygon" in obj and obj["segmentation_polygon"]:
                    # YOLO Seg Format: class x1 y1 x2 y2 ...
                    # Flatten the list of lists [[x,y], [x,y]] -> [x,y,x,y]
                    flat_poly = []
                    for poly_part in obj["segmentation_polygon"]:
                        if isinstance(poly_part, list):
                            flat_poly.extend(poly_part)
                        else:
                            flat_poly.append(poly_part)

                    # normalizing
                    normalized_points = []
                    for i in range(0, len(flat_poly), 2):
                        nx = flat_poly[i] / img_w
                        ny = flat_poly[i + 1] / img_h

                        # clamp to 0-1
                        nx = max(0.0, min(1.0, nx))
                        ny = max(0.0, min(1.0, ny))
                        normalized_points.append(f"{nx:.6f} {ny:.6f}")

                    line = f"{class_id} {' '.join(normalized_points)}\n"
                    f.write(line)

                else:
                    # Fallback to Bounding Box (YOLO Detect Format)
                    # class x_center y_center width height
                    x1, y1, x2, y2 = obj["bbox"]
                    w, h = x2 - x1, y2 - y1
                    x_c, y_c = x1 + w / 2, y1 + h / 2
                    line = f"{class_id} {x_c/img_w:.6f} {y_c/img_h:.6f} {w/img_w:.6f} {h/img_h:.6f}\n"
                    f.write(line)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Manual Refiner Tool for Mall Tracking Data"
    )
    # Explicit Arguments
    parser.add_argument(
        "--img_dir", type=str, help="Path to RAW images (e.g. ./data/cam_1)"
    )
    parser.add_argument(
        "--json_dir", type=str, help="Path to detected JSONs (e.g. ./output/cam_1/json)"
    )
    parser.add_argument("--out_dir", type=str, help="Path to save corrected dataset")

    # Legacy/Shortcuts
    parser.add_argument(
        "--cam", type=str, help="Shortcut: Camera ID (assumes ./data structure)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.img_dir and args.json_dir:
        img_path = Path(args.img_dir)
        json_path = Path(args.json_dir)

        if not img_path.exists():
            print(f"[ERROR] Image directory does not exist: {img_path}")
            sys.exit(1)
        if not json_path.exists():
            print(f"[ERROR] JSON directory does not exist: {json_path}")
            sys.exit(1)

        if args.out_dir:
            out_path = Path(args.out_dir)
        else:
            out_path = json_path.parent.parent / f"{json_path.parent.name}_corrected"

    elif args.cam:
        # Construct path from Camera ID
        cam_str = args.cam if args.cam.startswith("cam_") else f"cam_{args.cam}"
        img_path = Path(f"./data/{cam_str}")
        json_path = Path(f"./output_data/{cam_str}/json")
        out_path = Path(f"./output_data/{cam_str}_corrected")

    else:
        print("Error: You must provide either (--img_dir AND --json_dir) OR (--cam)")
        sys.exit(1)

    print(f"\n--- LAUNCHING REFINER ---")
    print(f"Images: {img_path}")
    print(f"JSONs:  {json_path}")
    print(f"Save:   {out_path}")
    print("-" * 30)

    # Launch GUI
    root = tk.Tk()
    # Center window fix
    root.eval("tk::PlaceWindow . center")
    app = MallTrackingRefiner(root, json_path, img_path, out_path)
    root.mainloop()
