import tkinter as tk
from tkinter import messagebox, ttk, filedialog
from PIL import Image, ImageTk
import json
import shutil
import argparse
import sys
from pathlib import Path
from omegaconf import OmegaConf
from types import SimpleNamespace


class MallTrackingRefiner:
    def __init__(self, root, json_dir, img_dir, out_dir, config_path) -> None:
        self.root = root
        self.cfg = self.load_config(config_path)
        self.fonts = SimpleNamespace(
            main=tuple(self.cfg.fonts.main),
            header=tuple(self.cfg.fonts.header),
            mono=tuple(self.cfg.fonts.mono),
            large_label=tuple(self.cfg.fonts.large_label),
        )
        self.root.title(f"{self.cfg.window.title_prefix} | {json_dir.name}")
        self.root.geometry(f"{self.cfg.window.width}x{self.cfg.window.height}")
        self.root.configure(bg=self.cfg.colors.bg)

        # Paths
        self.json_dir = json_dir
        self.img_dir = img_dir
        self.out_json = out_dir / "json"
        self.out_img = out_dir / "images"
        self.out_labels = out_dir / "labels"
        for p in [self.out_json, self.out_img, self.out_labels]:
            p.mkdir(parents=True, exist_ok=True)

        # Data Loading
        all_jsons = sorted(list(self.json_dir.glob("*.json")))
        self.files = [f for f in all_jsons if f.stem.isdigit()]

        if not self.files:
            messagebox.showerror("Error", "No numeric JSON files found!")
            sys.exit(1)

        self.current_idx = 0
        self.current_data = []
        self.selected_obj_idx = -1
        self.scale_factor = 1.0
        self.current_img_path = None

        # Interaction State
        self.drag_mode = None
        self.drag_start = None
        self.temp_rect = None
        self.handle_size = self.cfg.defaults.handle_size

        self.setup_ui()
        self.load_frame()
        self.bind_events()

    def load_config(self, path_str):
        path = Path(path_str)

        # 1. Existence Check
        if not path.exists():
            messagebox.showerror(
                "Configuration Error",
                f"Config file not found:\n{path}\n\n"
                "The tool strictly requires this file for styling.",
            )
            sys.exit(1)

        # 2. Load YAML
        try:
            cfg = OmegaConf.load(path)
        except Exception as e:
            messagebox.showerror("Configuration Error", f"Invalid YAML format:\n{e}")
            sys.exit(1)

        # 3. Validation (Fail Fast)
        # We check for a few critical keys to ensure the file isn't empty/corrupt
        required_keys = [
            "colors.highlight",
            "colors.moving",
            "colors.static",
            "window.width",
            "defaults.handle_size",
        ]

        missing = []
        for key in required_keys:
            # Navigate nested keys (e.g. colors -> highlight)
            parts = key.split(".")
            curr = cfg
            try:
                for p in parts:
                    curr = curr[p]
            except (KeyError, AttributeError):
                missing.append(key)

        if missing:
            msg = f"Config file is missing required keys:\n" + "\n".join(missing)
            print(f"[CRITICAL CONFIG ERROR] {msg}")
            messagebox.showerror("Configuration Error", msg)
            sys.exit(1)

        return cfg

    def setup_ui(self):
        c = self.cfg.colors

        # Top Bar (Progress)
        self.top_bar = tk.Frame(self.root, bg=c.panel_bg, height=40)
        self.top_bar.pack(side=tk.TOP, fill=tk.X)
        self.progress_var = tk.DoubleVar()
        ttk.Progressbar(
            self.top_bar, variable=self.progress_var, maximum=len(self.files)
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10, pady=10)
        self.lbl_progress = tk.Label(
            self.top_bar, text="0 / 0", bg=c.panel_bg, fg=c.fg, font=self.fonts.mono
        )
        self.lbl_progress.pack(side=tk.RIGHT, padx=10)

        # Main Container
        container = tk.Frame(self.root, bg=c.bg)
        container.pack(fill=tk.BOTH, expand=True)

        # Canvas (Left)
        self.canvas_frame = tk.Frame(container, bg=c.canvas_bg)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.canvas = tk.Canvas(
            self.canvas_frame, bg=c.canvas_bg, highlightthickness=0, cursor="crosshair"
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Sidebar (Right)
        self.sidebar = tk.Frame(container, width=320, bg=c.panel_bg)
        self.sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        self.sidebar.pack_propagate(False)

        # -- Controls --
        self.info_box = tk.Label(
            self.sidebar,
            text="Ready",
            bg="#333",
            fg=c.highlight,
            font=self.fonts.main,
            pady=5,
        )
        self.info_box.pack(fill=tk.X, pady=(0, 10))

        # Edit Group
        frm_sel = tk.LabelFrame(
            self.sidebar,
            text=" EDIT OBJECT ",
            bg=c.panel_bg,
            fg=c.highlight,
            font=self.fonts.header,
        )
        frm_sel.pack(fill=tk.X, padx=10, pady=5)

        # ID Row
        r1 = tk.Frame(frm_sel, bg=c.panel_bg)
        r1.pack(fill=tk.X, pady=5, padx=5)
        tk.Label(r1, text="ID (U):", bg=c.panel_bg, fg=c.fg).pack(side=tk.LEFT)
        self.ent_id = tk.Entry(
            r1, width=6, bg=c.entry_bg, fg=c.entry_fg, font=self.fonts.mono
        )
        self.ent_id.pack(side=tk.LEFT, padx=5)
        tk.Button(
            r1, text="SET", bg=c.btn_bg, fg=c.btn_fg, command=self.update_id
        ).pack(side=tk.LEFT)
        tk.Button(
            frm_sel,
            text="Propagate ID (P)",
            bg=c.btn_bg,
            fg=c.fg,
            command=self.propagate_id,
        ).pack(fill=tk.X, padx=5, pady=2)

        # Status Row
        r2 = tk.Frame(frm_sel, bg=c.panel_bg)
        r2.pack(fill=tk.X, pady=5, padx=5)
        tk.Label(r2, text="Status:", bg=c.panel_bg, fg=c.fg).pack(side=tk.LEFT)
        self.lbl_status = tk.Label(
            r2, text="-", width=8, bg="#333", fg="white", font=("Segoe UI", 10, "bold")
        )
        self.lbl_status.pack(side=tk.LEFT, padx=5)
        tk.Button(
            r2,
            text="Flip (M)",
            bg=c.btn_bg,
            fg=c.btn_fg,
            command=lambda: self.toggle_status(False),
        ).pack(side=tk.LEFT)
        tk.Button(
            frm_sel,
            text="Propagate Status (N)",
            bg=c.btn_bg,
            fg=c.fg,
            command=lambda: self.toggle_status(True),
        ).pack(fill=tk.X, padx=5, pady=2)
        tk.Button(
            frm_sel,
            text="DELETE (Del)",
            bg=c.moving,
            fg="white",
            font=self.fonts.header,
            command=self.delete_object,
        ).pack(fill=tk.X, padx=5, pady=10)

        # Nav
        frm_nav = tk.Frame(self.sidebar, bg=c.panel_bg)
        frm_nav.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        tk.Button(
            frm_nav,
            text="< Prev",
            bg=c.btn_bg,
            fg=c.fg,
            width=10,
            height=2,
            command=self.prev_frame,
        ).pack(side=tk.LEFT)
        tk.Button(
            frm_nav,
            text="Next >",
            bg="#2e8b57",
            fg="white",
            width=10,
            height=2,
            command=self.next_frame,
        ).pack(side=tk.RIGHT)

    def bind_events(self):
        self.root.bind("<Right>", lambda e: self.next_frame())
        self.root.bind("<Left>", lambda e: self.prev_frame())
        self.root.bind("<Shift-Right>", lambda e: self.jump_frame(50))
        self.root.bind("<Delete>", lambda e: self.delete_object())
        self.root.bind("m", lambda e: self.toggle_status(False))
        self.root.bind("n", lambda e: self.toggle_status(True))
        self.root.bind("p", lambda e: self.propagate_id())
        self.root.bind("u", lambda e: self.focus_id_entry())
        self.ent_id.bind("<Return>", lambda e: self.update_id())

        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

    def on_mouse_up(self, event):
        if not self.drag_mode:
            return

        if self.drag_mode == "create":
            x1, y1 = self.drag_start
            x2, y2 = event.x, event.y
            self.canvas.delete(self.temp_rect)

            # Normalize
            bx1, bx2 = sorted([x1, x2])
            by1, by2 = sorted([y1, y2])

            if (bx2 - bx1) > 10 and (by2 - by1) > 10:
                sf = self.scale_factor
                nx1, ny1, nx2, ny2 = bx1 / sf, by1 / sf, bx2 / sf, by2 / sf
                new_box = [nx1, ny1, nx2, ny2]

                # --- CRITICAL FIX: Create Polygon from Box ---
                # A rectangle polygon: TL -> TR -> BR -> BL
                # We save it as [[x,y], [x,y]...] (List of lists)
                box_poly = [[nx1, ny1], [nx2, ny1], [nx2, ny2], [nx1, ny2]]

                max_id = 0
                for o in self.current_data:
                    curr = int(o.get("global_id", 0))
                    if curr > max_id:
                        max_id = curr

                new_obj = {
                    "global_id": max_id + 1,
                    "local_id": -1,
                    "bbox": new_box,
                    "status": self.cfg.defaults.new_box_status,
                    "confidence": 1.0,
                    "segmentation_polygon": [box_poly],  # <--- SAVES POLYGON
                }
                self.current_data.append(new_obj)
                self.selected_obj_idx = len(self.current_data) - 1
                self.log(f"Created Object ID {max_id+1}", "success")

        elif self.selected_obj_idx >= 0 and self.drag_mode.startswith("resize"):
            # Commit resize logic (same as before)
            obj = self.current_data[self.selected_obj_idx]
            x1, y1, x2, y2 = [v * self.scale_factor for v in obj["bbox"]]
            cur_x, cur_y = event.x, event.y

            if self.drag_mode == "tl":
                x1, y1 = cur_x, cur_y
            elif self.drag_mode == "tr":
                x2, y1 = cur_x, cur_y
            elif self.drag_mode == "bl":
                x1, y2 = cur_x, cur_y
            elif self.drag_mode == "br":
                x2, y2 = cur_x, cur_y

            nx1, nx2 = sorted([x1, x2])
            ny1, ny2 = sorted([y1, y2])
            sf = self.scale_factor

            # Update Box
            obj["bbox"] = [nx1 / sf, ny1 / sf, nx2 / sf, ny2 / sf]

            # --- AUTO-UPDATE POLYGON ON RESIZE ---
            # If we resized the box, the old polygon is invalid.
            # Re-generate a box polygon so backprojection still works.
            new_poly = [
                [nx1 / sf, ny1 / sf],
                [nx2 / sf, ny1 / sf],
                [nx2 / sf, ny2 / sf],
                [nx1 / sf, ny2 / sf],
            ]
            obj["segmentation_polygon"] = [new_poly]

            self.log("Box Resized & Polygon Updated", "info")

        self.drag_mode = None
        self.redraw()

    def get_handle_at(self, x, y):
        """Check if mouse is over a resize handle of the selected object"""
        if self.selected_obj_idx < 0:
            return None

        # Get coords of selected box
        obj = self.current_data[self.selected_obj_idx]
        x1, y1, x2, y2 = [v * self.scale_factor for v in obj["bbox"]]

        s = self.handle_size + 2  # Tolerance

        # Check 4 corners
        if abs(x - x1) < s and abs(y - y1) < s:
            return "tl"  # Top-Left
        if abs(x - x2) < s and abs(y - y1) < s:
            return "tr"  # Top-Right
        if abs(x - x1) < s and abs(y - y2) < s:
            return "bl"  # Bottom-Left
        if abs(x - x2) < s and abs(y - y2) < s:
            return "br"  # Bottom-Right
        return None

    def on_mouse_down(self, event):
        x, y = event.x, event.y

        # 1. Check Resize Handles first
        handle = self.get_handle_at(x, y)
        if handle:
            self.drag_mode = handle
            self.drag_start = (x, y)
            return

        # 2. Check Creating New Box (Shift + Click)
        if event.state & 0x0001:  # Shift key held
            self.drag_mode = "create"
            self.drag_start = (x, y)
            self.temp_rect = self.canvas.create_rectangle(
                x, y, x, y, outline="green", width=2, dash=(4, 4)
            )
            return

        # 3. Standard Selection
        best_idx = -1
        min_area = float("inf")

        for idx, obj in enumerate(self.current_data):
            x1, y1, x2, y2 = [c * self.scale_factor for c in obj["bbox"]]
            if x1 <= x <= x2 and y1 <= y <= y2:
                area = (x2 - x1) * (y2 - y1)
                if area < min_area:
                    min_area = area
                    best_idx = idx

        self.selected_obj_idx = best_idx
        self.redraw()
        if best_idx >= 0:
            self.focus_id_entry()

    def on_mouse_drag(self, event):
        if not self.drag_mode:
            return
        cur_x, cur_y = event.x, event.y

        if self.drag_mode == "create":
            start_x, start_y = self.drag_start
            self.canvas.coords(self.temp_rect, start_x, start_y, cur_x, cur_y)

        elif self.selected_obj_idx >= 0:
            # Resizing logic
            obj = self.current_data[self.selected_obj_idx]
            x1, y1, x2, y2 = [v * self.scale_factor for v in obj["bbox"]]

            if self.drag_mode == "tl":
                x1, y1 = cur_x, cur_y
            elif self.drag_mode == "tr":
                x2, y1 = cur_x, cur_y
            elif self.drag_mode == "bl":
                x1, y2 = cur_x, cur_y
            elif self.drag_mode == "br":
                x2, y2 = cur_x, cur_y

            # Update canvas visual only (don't commit to data yet)
            # We convert back to native to save, but here we just redraw rect
            self.redraw(temp_override_coords=(x1, y1, x2, y2))

    def redraw(self, temp_override_coords=None):
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
        c = self.cfg.colors

        for idx, obj in enumerate(self.current_data):
            # 1. Determine Coordinates (Normal or Dragging override)
            if idx == self.selected_obj_idx and temp_override_coords:
                x1, y1, x2, y2 = temp_override_coords
            else:
                x1, y1, x2, y2 = [v * self.scale_factor for v in obj["bbox"]]

            status = obj.get("status", "Unknown")
            gid = obj.get("global_id", -1)

            # 2. Color Logic
            if idx == self.selected_obj_idx:
                outline, width, txt_c = c.highlight, 3, c.highlight
                # Draw Resize Handles
                s = self.cfg.defaults.handle_size
                for hx, hy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
                    self.canvas.create_rectangle(
                        hx - s, hy - s, hx + s, hy + s, fill="red"
                    )
            elif status == "Moving":
                outline, width, txt_c = c.moving, 2, c.moving
            elif status == "Static":
                outline, width, txt_c = c.static, 2, c.static
            elif status == "Analyzing":
                outline, width, txt_c = c.analyzing, 2, c.analyzing
            else:
                outline, width, txt_c = c.unknown, 2, c.unknown

            # 3. Draw Bounding Box
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=outline, width=width)
            self.canvas.create_text(
                x1,
                y1 - 15,
                text=f"{gid} ({status})",
                fill=txt_c,
                anchor=tk.SW,
                font=self.fonts.large_label,
            )

            # 4. Draw Polygons (THE FIX IS HERE)
            if "segmentation_polygon" in obj and obj["segmentation_polygon"]:
                for poly in obj["segmentation_polygon"]:
                    # --- ROBUST FLATTENING ---
                    flat_poly = []
                    if isinstance(poly, list):
                        for item in poly:
                            if isinstance(item, list):
                                flat_poly.extend(item)  # Flatten [x, y] -> x, y
                            else:
                                flat_poly.append(item)  # Already scalar

                    # Now we can safely multiply because it's guaranteed to be a list of numbers
                    scaled_poly = [v * self.scale_factor for v in flat_poly]

                    self.canvas.create_polygon(
                        scaled_poly, outline=outline, fill="", width=1, dash=(2, 2)
                    )

        self.update_sidebar()

    def update_sidebar(self):
        c = self.cfg.colors
        if self.selected_obj_idx >= 0:
            obj = self.current_data[self.selected_obj_idx]
            self.ent_id.delete(0, tk.END)
            self.ent_id.insert(0, str(obj.get("global_id", "")))

            stat = obj.get("status", "Unknown")
            self.lbl_status.config(text=stat)

            fg_col = c.unknown
            if stat == "Moving":
                fg_col = c.moving
            elif stat == "Static":
                fg_col = c.static
            elif stat == "Analyzing":
                fg_col = c.analyzing

            self.lbl_status.config(fg=fg_col)
        else:
            self.ent_id.delete(0, tk.END)
            self.lbl_status.config(text="-", fg=c.fg)

    def focus_id_entry(self):
        if self.selected_obj_idx >= 0:
            self.ent_id.focus_set()
            self.ent_id.selection_range(0, tk.END)

    def update_id(self):
        if self.selected_obj_idx >= 0:
            try:
                new_id = int(self.ent_id.get())
                self.current_data[self.selected_obj_idx]["global_id"] = new_id
                self.redraw()
                self.canvas.focus_set()  # Return focus to canvas for hotkeys
            except:
                pass

    def propagate_id(self):
        if self.selected_obj_idx < 0:
            return
        target = self.current_data[self.selected_obj_idx]
        local_id = target.get("local_id")
        try:
            new_gid = int(self.ent_id.get())
        except:
            return

        target["global_id"] = new_gid
        count = 0

        for i in range(self.current_idx + 1, len(self.files)):
            f_in = self.files[i]
            f_out = self.out_json / f_in.name
            load_p = f_out if f_out.exists() else f_in
            with open(load_p, "r") as f:
                data = json.load(f)

            found = False
            for obj in data:
                # If created manually, local_id might be -1, check global_id tracking?
                # Usually we track by local_id for robustness
                if obj.get("local_id") == local_id and local_id != -1:
                    obj["global_id"] = new_gid
                    found = True

            if found:
                with open(f_out, "w") as f:
                    json.dump(data, f, indent=2)
                count += 1

        self.log(f"Propagated ID {new_gid} to {count} frames", "success")
        self.redraw()

    def toggle_status(self, propagate=False):
        if self.selected_obj_idx < 0:
            return
        obj = self.current_data[self.selected_obj_idx]

        curr = obj.get("status", "Static")
        new = "Static" if curr == "Moving" else "Moving"
        if curr == "Unknown" or curr == "Analyzing":
            new = "Static"  # Default fallback

        obj["status"] = new
        msg = f"Set to {new}"

        if propagate:
            lid = obj.get("local_id")
            count = 0
            for i in range(self.current_idx + 1, len(self.files)):
                f_in = self.files[i]
                f_out = self.out_json / f_in.name
                load_p = f_out if f_out.exists() else f_in
                with open(load_p, "r") as f:
                    data = json.load(f)

                found = False
                for o in data:
                    if o.get("local_id") == lid and lid != -1:
                        o["status"] = new
                        found = True
                if found:
                    with open(f_out, "w") as f:
                        json.dump(data, f, indent=2)
                    count += 1
            msg = f"Propagated {new} to {count} frames"

        self.log(msg, "success")
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
            messagebox.showinfo("Info", "End of sequence")

    def prev_frame(self):
        self.save_current()
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_frame()

    def jump_frame(self, n):
        self.save_current()
        if self.current_idx + n < len(self.files):
            self.current_idx += n
            self.load_frame()

    def save_current(self):
        src_json = self.files[self.current_idx]
        dst_json = self.out_json / src_json.name
        with open(dst_json, "w") as f:
            json.dump(self.current_data, f, indent=2)

        if self.current_img_path:
            dst_img = self.out_img / self.current_img_path.name
            if not dst_img.exists():
                shutil.copy(self.current_img_path, dst_img)

        # Export YOLO Txt
        txt_path = self.out_labels / f"{src_json.stem}.txt"
        self.export_yolo_txt(txt_path, self.pil_img.width, self.pil_img.height)

    def export_yolo_txt(self, output_path, img_w, img_h):
        with open(output_path, "w") as f:
            for obj in self.current_data:
                # Class ID 0 for Person (Standard)
                class_id = 0

                # Check for Polygon first
                if "segmentation_polygon" in obj and obj["segmentation_polygon"]:
                    # --- ROBUST FLATTENING LOGIC ---
                    flat_poly = []
                    for poly_part in obj["segmentation_polygon"]:
                        if isinstance(poly_part, list):
                            # Handle potential nested lists (e.g. [[x,y], [x,y]] from manual tool)
                            for item in poly_part:
                                if isinstance(item, list):
                                    flat_poly.extend(item)  # Flatten [x, y] -> x, y
                                else:
                                    flat_poly.append(item)  # Already scalar
                        else:
                            flat_poly.append(poly_part)

                    # Now flat_poly is guaranteed to be [x1, y1, x2, y2, ...]
                    normalized_points = []
                    for i in range(0, len(flat_poly), 2):
                        # Safety check for index out of bounds
                        if i + 1 >= len(flat_poly):
                            break

                        nx = max(0.0, min(1.0, flat_poly[i] / img_w))
                        ny = max(0.0, min(1.0, flat_poly[i + 1] / img_h))
                        normalized_points.append(f"{nx:.6f} {ny:.6f}")

                    line = f"{class_id} {' '.join(normalized_points)}\n"
                    f.write(line)

                else:
                    # Fallback to Bounding Box (YOLO Detect Format)
                    x1, y1, x2, y2 = obj["bbox"]
                    w = x2 - x1
                    h = y2 - y1
                    x_c = x1 + (w / 2.0)
                    y_c = y1 + (h / 2.0)

                    # Normalize
                    x_c /= img_w
                    y_c /= img_h
                    w /= img_w
                    h /= img_h

                    # Clamp
                    x_c = max(0.0, min(1.0, x_c))
                    y_c = max(0.0, min(1.0, y_c))
                    w = max(0.0, min(1.0, w))
                    h = max(0.0, min(1.0, h))

                    line = f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n"
                    f.write(line)

    def log(self, msg, level="info"):
        col = self.cfg.colors.highlight if level == "info" else self.cfg.colors.static
        self.info_box.config(text=msg, fg=col)
        self.root.after(2000, lambda: self.info_box.config(text="Ready", fg="#888"))

    def load_frame(self):
        # (Same loading logic as V2: check output dir first)
        in_path = self.files[self.current_idx]
        out_path = self.out_json / in_path.name
        load_path = out_path if out_path.exists() else in_path

        with open(load_path, "r") as f:
            self.current_data = json.load(f)

        stem = in_path.stem
        for ext in [".jpg", ".png", ".jpeg"]:
            cand = self.img_dir / f"{stem}{ext}"
            if cand.exists():
                self.current_img_path = cand
                break

        if not self.current_img_path:
            messagebox.showerror("Error", f"Image missing for {stem}")
            return

        self.pil_img = Image.open(self.current_img_path)
        self.tk_img = ImageTk.PhotoImage(self.pil_img)
        self.selected_obj_idx = -1
        self.lbl_progress.config(text=f"{self.current_idx+1} / {len(self.files)}")
        self.progress_var.set(self.current_idx + 1)
        self.redraw()


if __name__ == "__main__":
    # Argument parsing logic...
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--json_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--config", type=str, default="../config/annotation_config.yml")
    args = parser.parse_args()

    print(f"\n" + "=" * 40)
    print(f"   REFINER PRO (By Soumo Roy)")
    print(f"=" * 40)
    print(f"Config: {args.config}")
    print(f"Images: {args.img_dir}")
    print(f"JSONs:  {args.json_dir}")
    print(f"Output: {args.out_dir}")
    print(f"-" * 40)

    img_path, json_path = Path(args.img_dir), Path(args.json_dir)
    out_path = (
        Path(args.out_dir)
        if args.out_dir
        else json_path.parent / f"{json_path.name}_corrected"
    )

    root = tk.Tk()
    app = MallTrackingRefiner(root, json_path, img_path, out_path, args.config)
    root.mainloop()
