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

        screen_w = self.root.winfo_width()
        screen_h = self.root.winfo_height()
