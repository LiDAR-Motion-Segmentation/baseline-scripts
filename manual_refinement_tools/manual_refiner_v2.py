import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import json
import shutil
import argparse
import sys
from pathlib import Path
from omegaconf import OmegaConf
from types import SimpleNamespace


class MallTrackingRefiner:
    def __init__(self, root, json_dir, img_dir, out_dir, ui_cfg):
        self.root = root
        self.cfg = ui_cfg

        self.root.title(f"{self.cfg.window.title_prefix} | {json_dir.name}")
        self.root.geometry(f"{self.cfg.window.width}x{self.cfg.window.height}")
        self.root.configure(bg=self.cfg.colors.background)

        self.fonts = SimpleNamespace(
            main=tuple(self.cfg.fonts.main),
            header=tuple(self.cfg.fonts.header),
            mono=tuple(self.cfg.fonts.mono),
            large_label=tuple(self.cfg.fonts.large_label),
        )

        self.json_dir = json_dir
        self.img_dir = img_dir
        self.out_json = out_dir / "json"
        self.out_img = out_dir / "images"
        self.out_labels = out_dir / "labels"

        for p in [self.out_json, self.out_img, self.out_labels]:
            p.mkdir(parents=True, exist_ok=True)

        all_jsons = sorted(list(self.json_dir.glob("*.json")))
        self.files = [f for f in all_jsons if f.stem.isdigit()]

        if not self.files:
            messagebox.showerror("Error", f"No numeric JSONs found!")
            sys.exit(1)

        self.current_idx = 0
        self.current_data = []
        self.selected_obj_idx = -1
        self.scale_factor = 1.0
        self.current_img_path = None

        self.setup_ui()
        self.load_frame()
        self.bind_hotkeys()

    def setup_ui(self):
        c = self.cfg.colors  # Short alias for cleaner code
        f = self.fonts
        self.top_bar = tk.Frame(self.root, bg=c.panel_bg, height=40)
        self.top_bar.pack(side=tk.TOP, fill=tk.X)

        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(
            self.top_bar, variable=self.progress_var, maximum=len(self.files)
        )
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10, pady=10)

        self.lbl_progress = tk.Label(
            self.top_bar, text="0 / 0", bg=c.panel_bg, fg=c.text_primary, font=f.mono
        )
        self.lbl_progress.pack(side=tk.RIGHT, padx=10)

        self.container = tk.Frame(self.root, bg=c.background)
        self.container.pack(fill=tk.BOTH, expand=True)

        # LEFT: Canvas
        self.canvas_frame = tk.Frame(self.container, bg=c.canvas_bg)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.canvas = tk.Canvas(self.canvas_frame, bg=c.canvas_bg, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # RIGHT: Sidebar
        self.sidebar = tk.Frame(self.container, width=350, bg=c.panel_bg)
        self.sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        self.sidebar.pack_propagate(False)

        # Info Box
        self.info_box = tk.Label(
            self.sidebar, text="Ready", bg="#333", fg=c.accent, font=f.main, pady=5
        )
        self.info_box.pack(fill=tk.X, pady=(0, 10))

        # Selection Panel
        self.frm_sel = tk.LabelFrame(
            self.sidebar,
            text=" ACTIVE SELECTION ",
            bg=c.panel_bg,
            fg=c.highlight,
            font=f.header,
        )
        self.frm_sel.pack(fill=tk.X, padx=10, pady=5)

        # ID Row
        frm_id = tk.Frame(self.frm_sel, bg=c.panel_bg)
        frm_id.pack(fill=tk.X, pady=5, padx=5)
        tk.Label(frm_id, text="Global ID:", bg=c.panel_bg, fg=c.text_primary).pack(
            side=tk.LEFT
        )
        self.ent_id = tk.Entry(
            frm_id,
            width=8,
            bg=c.entry_bg,
            fg=c.entry_fg,
            insertbackground="white",
            font=f.mono,
        )
        self.ent_id.pack(side=tk.LEFT, padx=5)
        tk.Button(
            frm_id,
            text="SET",
            bg=c.accent,
            fg="white",
            relief="flat",
            command=self.update_id,
        ).pack(side=tk.LEFT)

        # Propagate ID
        tk.Button(
            self.frm_sel,
            text=" Propagate ID Forward (P)",
            bg=c.button_bg,
            fg=c.text_primary,
            relief="flat",
            command=self.propagate_id,
        ).pack(fill=tk.X, padx=5, pady=5)

        # Status Row
        frm_stat = tk.Frame(self.frm_sel, bg=c.panel_bg)
        frm_stat.pack(fill=tk.X, pady=5, padx=5)
        tk.Label(frm_stat, text="Status:", bg=c.panel_bg, fg=c.text_primary).pack(
            side=tk.LEFT
        )
        self.lbl_status = tk.Label(
            frm_stat, text="-", width=8, bg="#333", fg="white", font=f.header
        )
        self.lbl_status.pack(side=tk.LEFT, padx=5)
        tk.Button(
            frm_stat,
            text=" FLIP (M)",
            bg="#555",
            fg="white",
            relief="flat",
            command=lambda: self.toggle_status(False),
        ).pack(side=tk.LEFT)

        # Propagate Status
        tk.Button(
            self.frm_sel,
            text=" Propagate Status Fwd (Shift+M)",
            bg=c.button_bg,
            fg=c.text_primary,
            relief="flat",
            command=lambda: self.toggle_status(True),
        ).pack(fill=tk.X, padx=5, pady=5)

        # Delete
        tk.Button(
            self.frm_sel,
            text=" DELETE OBJECT (Del)",
            bg=c.danger,
            fg="white",
            font=f.header,
            relief="flat",
            command=self.delete_object,
        ).pack(fill=tk.X, padx=5, pady=10)

        # Navigation Panel
        tk.Label(
            self.sidebar,
            text=" NAVIGATION ",
            bg=c.panel_bg,
            fg=c.text_secondary,
            font=f.header,
        ).pack(pady=(20, 5))

        frm_nav = tk.Frame(self.sidebar, bg=c.panel_bg)
        frm_nav.pack(fill=tk.X, padx=10)

        tk.Button(
            frm_nav,
            text="< Prev",
            bg=c.button_bg,
            fg=c.text_primary,
            height=2,
            width=10,
            relief="flat",
            command=self.prev_frame,
        ).pack(side=tk.LEFT)
        tk.Button(
            frm_nav,
            text="Next >",
            bg=c.success,
            fg="white",
            height=2,
            width=10,
            relief="flat",
            command=self.next_frame,
        ).pack(side=tk.RIGHT)

        # Jump Buttons
        frm_jump = tk.Frame(self.sidebar, bg=c.panel_bg)
        frm_jump.pack(fill=tk.X, padx=10, pady=10)
        tk.Button(
            frm_jump,
            text="+50",
            bg="#444",
            fg="#aaa",
            relief="flat",
            command=lambda: self.jump_frame(50),
        ).pack(side=tk.RIGHT, padx=2)
        tk.Button(
            frm_jump,
            text="+10",
            bg="#444",
            fg="#aaa",
            relief="flat",
            command=lambda: self.jump_frame(10),
        ).pack(side=tk.RIGHT, padx=2)
        tk.Label(frm_jump, text="Jump:", bg=c.panel_bg, fg=c.text_secondary).pack(
            side=tk.RIGHT
        )

    def bind_hotkeys(self):
        self.root.bind("<Right>", lambda e: self.next_frame())
        self.root.bind("<Left>", lambda e: self.prev_frame())
        self.root.bind("<Shift-Right>", lambda e: self.jump_frame(50))
        self.root.bind("<Control-Right>", lambda e: self.jump_frame(100))
        self.root.bind("<Delete>", lambda e: self.delete_object())
        self.root.bind("m", lambda e: self.toggle_status(propagate=False))
        self.root.bind("M", lambda e: self.toggle_status(propagate=True))
        self.root.bind("p", lambda e: self.propagate_id())

    def log(self, message, level="info"):
        color = self.cfg.colors.accent if level == "info" else self.cfg.colors.success
        self.info_box.config(text=message, fg=color)
        self.root.after(
            3000, lambda: self.info_box.config(text="Ready", fg=self.cfg.colors.accent)
        )

    def load_frame(self):
        input_json_path = self.files[self.current_idx]
        output_json_path = self.out_json / input_json_path.name
        load_path = output_json_path if output_json_path.exists() else input_json_path

        with open(load_path, "r") as f:
            self.current_data = json.load(f)

        stem = input_json_path.stem
        valid_exts = [".png", ".jpg", ".jpeg", ".bmp"]
        self.current_img_path = None
        for ext in valid_exts:
            cand = self.img_dir / f"{stem}{ext}"
            if cand.exists():
                self.current_img_path = cand
                break

        if not self.current_img_path:
            messagebox.showerror("Error", f"Image not found for {stem}")
            return

        self.pil_img = Image.open(self.current_img_path)
        self.scale_factor = 1.0
        self.tk_img = ImageTk.PhotoImage(self.pil_img)

        self.selected_obj_idx = -1

        self.lbl_progress.config(text=f"{self.current_idx + 1} / {len(self.files)}")
        self.progress_var.set(self.current_idx + 1)
        self.root.title(f"{self.cfg.window.title_prefix} | {input_json_path.name}")

        self.redraw()

    def redraw(self):
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

        c = self.cfg.colors  # Alias

        for idx, obj in enumerate(self.current_data):
            x1, y1, x2, y2 = [c * self.scale_factor for c in obj["bbox"]]
            gid = obj.get("global_id", -1)
            status = obj.get("status", "Unknown")

            if idx == self.selected_obj_idx:
                outline, width, txt_c = c.highlight, 3, c.highlight
            elif status == "Moving":
                outline, width, txt_c = c.danger, 2, c.danger
            else:
                outline, width, txt_c = c.accent, 2, c.accent

            self.canvas.create_rectangle(x1, y1, x2, y2, outline=outline, width=width)
            self.canvas.create_text(
                x1,
                y1 - 15,
                text=f"{gid}",
                fill=txt_c,
                anchor=tk.SW,
                font=self.fonts.large_label,
            )

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

        for idx, obj in enumerate(self.current_data):
            x1, y1, x2, y2 = [c * self.scale_factor for c in obj["bbox"]]
            if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                area = (x2 - x1) * (y2 - y1)
                if area < min_area:
                    min_area = area
                    best_idx = idx

        self.selected_obj_idx = best_idx
        self.redraw()
        if best_idx >= 0:
            self.ent_id.focus_set()
            self.ent_id.selection_range(0, tk.END)

    def update_control_panel(self):
        c = self.cfg.colors
        if self.selected_obj_idx >= 0:
            obj = self.current_data[self.selected_obj_idx]
            self.ent_id.delete(0, tk.END)
            self.ent_id.insert(0, str(obj.get("global_id", "")))

            stat = obj.get("status", "Unknown")
            self.lbl_status.config(text=stat)

            if stat == "Moving":
                self.lbl_status.config(fg=c.danger)
            else:
                self.lbl_status.config(fg=c.accent)

            self.frm_sel.config(
                text=f" OBJECT {self.selected_obj_idx} SELECTED ", fg=c.highlight
            )
        else:
            self.ent_id.delete(0, tk.END)
            self.lbl_status.config(text="-", fg="white")
            self.frm_sel.config(text=" NO SELECTION ", fg=c.text_secondary)

    def update_id(self):
        if self.selected_obj_idx >= 0:
            try:
                new_id = int(self.ent_id.get())
                self.current_data[self.selected_obj_idx]["global_id"] = new_id
                self.log(f"Updated ID to {new_id}")
                self.redraw()
                self.canvas.focus_set()
            except ValueError:
                pass

    def propagate_id(self):
        if self.selected_obj_idx < 0:
            return
        target_obj = self.current_data[self.selected_obj_idx]
        local_id_to_track = target_obj.get("local_id")
        try:
            new_global_id = int(self.ent_id.get())
        except ValueError:
            return

        target_obj["global_id"] = new_global_id
        count = 0
        for i in range(self.current_idx + 1, len(self.files)):
            f_in = self.files[i]
            f_out = self.out_json / f_in.name
            load_p = f_out if f_out.exists() else f_in
            with open(load_p, "r") as f:
                data = json.load(f)

            found = False
            for obj in data:
                if obj.get("local_id") == local_id_to_track:
                    obj["global_id"] = new_global_id
                    found = True
            if found:
                with open(f_out, "w") as f:
                    json.dump(data, f, indent=2)
                count += 1

        self.log(f"Propagated ID {new_global_id} to {count} frames!", "success")
        self.redraw()

    def toggle_status(self, propagate=False):
        if self.selected_obj_idx < 0:
            return
        target_obj = self.current_data[self.selected_obj_idx]
        curr = target_obj.get("status", "Static")
        new_status = "Static" if curr == "Moving" else "Moving"
        target_obj["status"] = new_status

        msg = f"Flipped to {new_status}"

        if propagate:
            local_id_to_track = target_obj.get("local_id")
            count = 0
            for i in range(self.current_idx + 1, len(self.files)):
                f_in = self.files[i]
                f_out = self.out_json / f_in.name
                load_p = f_out if f_out.exists() else f_in
                with open(load_p, "r") as f:
                    data = json.load(f)

                found = False
                for obj in data:
                    if obj.get("local_id") == local_id_to_track:
                        obj["status"] = new_status
                        found = True
                if found:
                    with open(f_out, "w") as f:
                        json.dump(data, f, indent=2)
                    count += 1
            msg = f"Propagated {new_status} to {count} frames!"

        self.log(msg, "success")
        self.redraw()

    def delete_object(self):
        if self.selected_obj_idx >= 0:
            del self.current_data[self.selected_obj_idx]
            self.selected_obj_idx = -1
            self.log("Object Deleted", "success")
            self.redraw()

    def next_frame(self):
        self.save_current()
        if self.current_idx < len(self.files) - 1:
            self.current_idx += 1
            self.load_frame()
        else:
            messagebox.showinfo("Done", "End of sequence!")

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
        else:
            self.log("Cannot jump past end")

    def save_current(self):
        src_json = self.files[self.current_idx]
        dst_json = self.out_json / src_json.name
        with open(dst_json, "w") as f:
            json.dump(self.current_data, f, indent=2)

        if self.current_img_path and self.current_img_path.exists():
            dst_img = self.out_img / self.current_img_path.name
            if not dst_img.exists():
                shutil.copy(self.current_img_path, dst_img)

        if hasattr(self, "pil_img"):
            txt_filename = f"{src_json.stem}.txt"
            dst_txt = self.out_labels / txt_filename
            self.export_yolo_txt(dst_txt, self.pil_img.width, self.pil_img.height)

    def export_yolo_txt(self, output_path, img_w, img_h):
        with open(output_path, "w") as f:
            for obj in self.current_data:
                class_id = 0
                if "segmentation_polygon" in obj and obj["segmentation_polygon"]:
                    flat_poly = []
                    for poly_part in obj["segmentation_polygon"]:
                        if isinstance(poly_part, list):
                            flat_poly.extend(poly_part)
                        else:
                            flat_poly.append(poly_part)
                    normalized_points = []
                    for i in range(0, len(flat_poly), 2):
                        nx = max(0.0, min(1.0, flat_poly[i] / img_w))
                        ny = max(0.0, min(1.0, flat_poly[i + 1] / img_h))
                        normalized_points.append(f"{nx:.6f} {ny:.6f}")
                    line = f"{class_id} {' '.join(normalized_points)}\n"
                    f.write(line)
                else:
                    x1, y1, x2, y2 = obj["bbox"]
                    w, h = x2 - x1, y2 - y1
                    x_c, y_c = x1 + w / 2, y1 + h / 2
                    line = f"{class_id} {x_c/img_w:.6f} {y_c/img_h:.6f} {w/img_w:.6f} {h/img_h:.6f}\n"
                    f.write(line)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--json_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--config", type=str, default="conf/ui_theme.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    img_path, json_path = Path(args.img_dir), Path(args.json_dir)
    out_path = (
        Path(args.out_dir)
        if args.out_dir
        else json_path.parent / f"{json_path.name}_corrected"
    )

    # --- STRICT CONFIG LOADING ---
    cfg_path = Path(args.config)

    if not cfg_path.exists():
        print(f"\n[CRITICAL ERROR] UI Config file not found: {cfg_path}")
        print(f"The tool requires a theme configuration file to run.")
        print(
            f"Please ensure 'conf/ui_theme.yaml' exists or pass a path via --config\n"
        )
        sys.exit(1)

    try:
        ui_cfg = OmegaConf.load(cfg_path)
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Failed to parse config file: {cfg_path}")
        print(f"Error details: {e}\n")
        sys.exit(1)

    # --- DATA PATH VALIDATION ---
    if not img_path.exists():
        print(f"[ERROR] Image directory does not exist: {img_path}")
        sys.exit(1)

    if not json_path.exists():
        print(f"[ERROR] JSON directory does not exist: {json_path}")
        sys.exit(1)

    print(f"\n" + "=" * 40)
    print(f"   REFINER PRO (Strict Mode)")
    print(f"=" * 40)
    print(f"Config: {cfg_path}")
    print(f"Images: {img_path}")
    print(f"JSONs:  {json_path}")
    print(f"Output: {out_path}")
    print(f"-" * 40)

    root = tk.Tk()
    root.eval("tk::PlaceWindow . center")
    app = MallTrackingRefiner(root, json_path, img_path, out_path, ui_cfg)
    root.mainloop()
