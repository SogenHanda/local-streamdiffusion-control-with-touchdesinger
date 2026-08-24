from __future__ import annotations

import os
import queue
import subprocess
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

from PIL import Image, ImageTk

from .config import AppConfig, DEFAULT_CONFIG_PATH
from .model_catalog import (
    PERFORMANCE_PRESETS,
    ModelProfile,
    discover_model_profiles,
    find_performance_preset,
    find_profile,
    infer_performance_preset,
)
from .osc_control import (
    OSC_CONFIG_DEBOUNCE_MS,
    OSC_HOST,
    OSC_PORT,
    OSCCommand,
    OSCEvent,
    OSCMessage,
    OSCUDPServer,
    apply_osc_config,
    decode_control_message,
)
from .runtime import DiffusionWorker, WorkerEvent
from .tensorrt_backend import inspect_tensorrt


BACKGROUND = "#11151c"
PANEL = "#1a202b"
PANEL_ALT = "#222a37"
TEXT = "#eef3f8"
MUTED = "#9aa8b8"
ACCENT = "#69d2a3"
WARNING = "#f2c14e"
ERROR = "#ef6a6a"
BACKEND_LABELS = {
    "Auto（TensorRT優先）": "auto",
    "TensorRT FP16": "tensorrt",
    "xFormers": "xformers",
    "PyTorch": "pytorch",
}
OSC_MONITOR_GROUPS = (
    (
        "LIVE / 即時反映",
        (
            "/ergonomics/live/prompt",
            "/ergonomics/live/strength",
            "/ergonomics/live/seed",
            "/ergonomics/live/target_fps",
            "/ergonomics/live/input_feedback",
            "/ergonomics/live/output_smoothing",
            "/ergonomics/live/latent_morph",
            "/ergonomics/live/history_frames",
            "/ergonomics/live/motion_threshold",
        ),
    ),
    (
        "CONFIG / 再初期化",
        (
            "/ergonomics/config/model_index",
            "/ergonomics/config/backend_index",
            "/ergonomics/config/width",
            "/ergonomics/config/height",
            "/ergonomics/config/performance_index",
            "/ergonomics/config/spout_input",
            "/ergonomics/config/spout_output",
            "/ergonomics/config/spout_sample_fps",
            "/ergonomics/config/apply",
        ),
    ),
    (
        "SYSTEM / 操作",
        (
            "/ergonomics/system/start",
            "/ergonomics/system/stop",
            "/ergonomics/temporal/reset",
        ),
    ),
)
OSC_TRIGGER_ADDRESSES = {
    "/ergonomics/config/apply",
    "/ergonomics/system/start",
    "/ergonomics/system/stop",
    "/ergonomics/temporal/reset",
}


class DiffusionApp(tk.Tk):
    def __init__(
        self,
        config_path: Path = DEFAULT_CONFIG_PATH,
        *,
        autostart: bool = False,
    ) -> None:
        super().__init__()
        self.title("Ergonomics Local Diffusion Monitor")
        self.geometry("1400x960")
        self.minsize(1160, 800)
        self.configure(bg=BACKGROUND)

        self.config_path = config_path
        self.events: "queue.Queue[WorkerEvent]" = queue.Queue(maxsize=16)
        self.worker = DiffusionWorker(self.events)
        self._preview_refs: dict[str, ImageTk.PhotoImage] = {}
        self._closing = False
        self._state = "停止"
        self._live_update_after_id: str | None = None
        self._trt_build_process: subprocess.Popen[str] | None = None
        self.osc_events: "queue.Queue[OSCEvent]" = queue.Queue(maxsize=256)
        self.osc_server: OSCUDPServer | None = None
        self._osc_pending_config: dict[str, Any] = {}
        self._osc_config_after_id: str | None = None
        self._osc_restart_config: AppConfig | None = None
        self._osc_restart_after_id: str | None = None
        self._osc_last_live_values: dict[str, Any] = {}
        self._osc_last_config_values: dict[str, Any] = {}
        self._osc_trigger_active: dict[str, bool] = {}
        self._generation_desired = False

        try:
            config = AppConfig.load(config_path)
        except Exception as exc:
            config = AppConfig()
            messagebox.showwarning("設定読込エラー", f"既定設定で起動します。\n\n{exc}")

        self._configure_styles()
        self._create_variables(config)
        self._build_monitor_only_ui()
        self._set_controls_for_state("停止")
        self._start_osc_server()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(80, self._poll_events)
        if autostart:
            # Let Tk finish drawing before model loading starts. This also makes
            # startup failures visible in the normal application window.
            self.after(600, self._start)

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=BACKGROUND)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("PanelAlt.TFrame", background=PANEL_ALT)
        style.configure("TLabel", background=BACKGROUND, foreground=TEXT, font=("Yu Gothic UI", 10))
        style.configure("Panel.TLabel", background=PANEL, foreground=TEXT, font=("Yu Gothic UI", 10))
        style.configure("Muted.Panel.TLabel", background=PANEL, foreground=MUTED, font=("Yu Gothic UI", 9))
        style.configure("Title.TLabel", background=BACKGROUND, foreground=TEXT, font=("Yu Gothic UI Semibold", 20))
        style.configure("Subtitle.TLabel", background=BACKGROUND, foreground=MUTED, font=("Yu Gothic UI", 10))
        style.configure("Section.Panel.TLabel", background=PANEL, foreground=TEXT, font=("Yu Gothic UI Semibold", 12))
        style.configure("MetricName.TLabel", background=PANEL_ALT, foreground=MUTED, font=("Yu Gothic UI", 9))
        style.configure("MetricValue.TLabel", background=PANEL_ALT, foreground=TEXT, font=("Consolas", 13, "bold"))
        style.configure("TEntry", fieldbackground="#0e1218", foreground=TEXT, insertcolor=TEXT, bordercolor="#364152")
        style.configure("TCombobox", fieldbackground="#0e1218", foreground=TEXT, arrowcolor=TEXT)
        style.map("TCombobox", fieldbackground=[("readonly", "#0e1218")], foreground=[("readonly", TEXT)])
        style.configure("TCheckbutton", background=PANEL, foreground=TEXT, font=("Yu Gothic UI", 9))
        style.map("TCheckbutton", background=[("active", PANEL)], foreground=[("active", TEXT)])
        style.configure("Accent.TButton", background=ACCENT, foreground="#0b1612", font=("Yu Gothic UI Semibold", 10), padding=(12, 8))
        style.map("Accent.TButton", background=[("active", "#86e4ba"), ("disabled", "#43554d")])
        style.configure("TButton", background="#303b4b", foreground=TEXT, font=("Yu Gothic UI", 10), padding=(10, 7))
        style.map("TButton", background=[("active", "#3b485b"), ("disabled", "#222a34")], foreground=[("disabled", "#697585")])
        style.configure("Horizontal.TScale", background=PANEL, troughcolor="#0e1218")

    def _create_variables(self, config: AppConfig) -> None:
        self.model_profiles = discover_model_profiles()
        active_profile = find_profile(
            self.model_profiles,
            config.model_profile,
            config.model_path,
        )
        self._profile_by_display = {
            profile.display_label: profile for profile in self.model_profiles
        }
        self.model_profile_var = tk.StringVar(
            value=active_profile.display_label if active_profile else "カスタムパス"
        )
        self.model_description_var = tk.StringVar(
            value=active_profile.description if active_profile else "モデルパスを直接指定します。"
        )
        self.model_path_var = tk.StringVar(value=config.model_path)
        self.lcm_lora_path_var = tk.StringVar(value=config.lcm_lora_path)
        self.spout_input_var = tk.StringVar(value=config.spout_input)
        self.spout_output_var = tk.StringVar(value=config.spout_output)
        self.resolution_var = tk.StringVar(value=f"{config.width} × {config.height}")
        self.denoise_var = tk.IntVar(value=config.denoise_index)
        self.denoise_label_var = tk.StringVar()
        self.seed_var = tk.IntVar(value=config.seed)
        self.target_fps_var = tk.DoubleVar(value=config.target_fps)
        self.spout_sample_fps_var = tk.DoubleVar(value=config.spout_sample_fps)
        backend_label = next(
            (
                label
                for label, value in BACKEND_LABELS.items()
                if value == config.acceleration_backend
            ),
            "Auto（TensorRT優先）",
        )
        self.acceleration_backend_var = tk.StringVar(value=backend_label)
        self.tensorrt_status_var = tk.StringVar(value="状態を確認しています…")
        self.use_lcm_lora_var = tk.BooleanVar(value=config.use_lcm_lora)
        self.lcm_steps_var = tk.StringVar(
            value="1 step（高速）" if config.lcm_steps == 1 else "2 steps（高品質）"
        )
        active_performance = infer_performance_preset(
            config.lcm_steps,
            config.use_tiny_vae,
        )
        self.performance_preset_var = tk.StringVar(value=active_performance.label)
        self.performance_description_var = tk.StringVar(value=active_performance.description)
        self.tiny_vae_var = tk.BooleanVar(value=config.use_tiny_vae)
        self.temporal_feedback_var = tk.DoubleVar(value=config.temporal_feedback)
        self.temporal_smoothing_var = tk.DoubleVar(value=config.temporal_smoothing)
        self.latent_morph_strength_var = tk.DoubleVar(value=config.latent_morph_strength)
        self.latent_history_frames_var = tk.DoubleVar(value=config.latent_history_frames)
        self.scene_cut_threshold_var = tk.DoubleVar(value=config.scene_cut_threshold)
        self.temporal_feedback_label_var = tk.StringVar()
        self.temporal_smoothing_label_var = tk.StringVar()
        self.latent_morph_label_var = tk.StringVar()
        self.latent_history_label_var = tk.StringVar()
        self.scene_cut_label_var = tk.StringVar()
        self.flip_input_var = tk.BooleanVar(value=config.flip_input)
        self.flip_output_var = tk.BooleanVar(value=config.flip_output)
        self.offline_var = tk.BooleanVar(value=config.offline_mode)
        self.status_var = tk.StringVar(value="停止")
        self.status_detail_var = tk.StringVar(
            value="TouchDesignerからのOSC Startを待っています。"
        )
        self.denoise_var.trace_add("write", self._on_denoise_changed)
        self.seed_var.trace_add("write", self._on_live_value_changed)
        self.target_fps_var.trace_add("write", self._on_live_value_changed)
        self.temporal_feedback_var.trace_add("write", self._on_temporal_changed)
        self.temporal_smoothing_var.trace_add("write", self._on_temporal_changed)
        self.latent_morph_strength_var.trace_add("write", self._on_temporal_changed)
        self.latent_history_frames_var.trace_add("write", self._on_temporal_changed)
        self.scene_cut_threshold_var.trace_add("write", self._on_temporal_changed)
        self._update_denoise_label()
        self._update_temporal_labels()
        self._initial_prompt = config.prompt

    def _build_monitor_only_ui(self) -> None:
        """Build a read-only operation screen controlled entirely by OSC."""
        header = ttk.Frame(self, padding=(24, 18, 24, 12))
        header.pack(fill="x")
        ttk.Label(
            header,
            text="LOCAL DIFFUSION MONITOR",
            style="Title.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            header,
            text=(
                "TouchDesigner / OSC control / Spout / StreamDiffusion — "
                "Python側は監視専用"
            ),
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(3, 0))

        body = ttk.Frame(self, padding=(20, 0, 20, 18))
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=0, minsize=430)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        osc_panel = ttk.Frame(body, style="Panel.TFrame")
        osc_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        self._build_osc_monitor(osc_panel)

        monitor = ttk.Frame(body, style="Panel.TFrame", padding=18)
        monitor.grid(row=0, column=1, sticky="nsew")
        monitor.columnconfigure(0, weight=1)
        monitor.rowconfigure(3, weight=1)
        self._build_monitor(monitor)

        # Kept for configuration collection and programmatic OSC reflection;
        # no interactive parameter widgets are created in monitoring mode.
        self.restart_controls: list[tk.Widget] = []

    def _build_osc_monitor(self, parent: ttk.Frame) -> None:
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)
        canvas = tk.Canvas(
            parent,
            bg=PANEL,
            highlightthickness=0,
            width=410,
        )
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=scrollbar.set)

        content = ttk.Frame(canvas, style="Panel.TFrame", padding=16)
        window = canvas.create_window((0, 0), window=content, anchor="nw")
        content.columnconfigure(0, weight=1)
        content.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.bind(
            "<Configure>",
            lambda event: canvas.itemconfigure(window, width=event.width),
        )
        scroll = lambda event: canvas.yview_scroll(
            -1 if event.delta > 0 else 1,
            "units",
        )
        canvas.bind("<Enter>", lambda _event: self.bind_all("<MouseWheel>", scroll))
        canvas.bind("<Leave>", lambda _event: self.unbind_all("<MouseWheel>"))

        ttk.Label(
            content,
            text="OSC CONTROL MONITOR",
            style="Section.Panel.TLabel",
        ).grid(row=0, column=0, sticky="w")
        self.osc_listener_var = tk.StringVar(value=f"待受準備中: {OSC_HOST}:{OSC_PORT}")
        self.osc_last_var = tk.StringVar(value="最終受信: -")
        self.osc_count_var = tk.StringVar(value="受信数: 0")
        ttk.Label(
            content,
            textvariable=self.osc_listener_var,
            style="Muted.Panel.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Label(
            content,
            textvariable=self.osc_last_var,
            style="Muted.Panel.TLabel",
            wraplength=370,
        ).grid(row=2, column=0, sticky="w", pady=(2, 0))
        ttk.Label(
            content,
            textvariable=self.osc_count_var,
            style="Muted.Panel.TLabel",
        ).grid(row=3, column=0, sticky="w", pady=(2, 10))

        ttk.Label(
            content,
            text="現在適用中の設定",
            style="Section.Panel.TLabel",
        ).grid(row=4, column=0, sticky="w", pady=(4, 5))
        self.effective_setting_vars = {
            key: tk.StringVar(value="-")
            for key in (
                "model",
                "backend",
                "performance",
                "resolution",
                "spout",
                "generation",
                "temporal",
            )
        }
        effective_labels = (
            ("model", "MODEL"),
            ("backend", "BACKEND"),
            ("performance", "PERFORMANCE"),
            ("resolution", "RESOLUTION"),
            ("spout", "SPOUT I/O"),
            ("generation", "GENERATION"),
            ("temporal", "TEMPORAL"),
        )
        row = 5
        for key, label in effective_labels:
            card = ttk.Frame(content, style="PanelAlt.TFrame", padding=(9, 6))
            card.grid(row=row, column=0, sticky="ew", pady=2)
            ttk.Label(card, text=label, style="MetricName.TLabel").pack(anchor="w")
            ttk.Label(
                card,
                textvariable=self.effective_setting_vars[key],
                style="Panel.TLabel",
                wraplength=360,
            ).pack(anchor="w", pady=(2, 0))
            row += 1

        ttk.Label(
            content,
            text="PROMPT（OSC反映値・読取専用）",
            style="MetricName.TLabel",
        ).grid(row=row, column=0, sticky="w", pady=(7, 3))
        self.prompt_text = tk.Text(
            content,
            height=4,
            wrap="word",
            state="disabled",
            bg="#0e1218",
            fg=TEXT,
            relief="flat",
            padx=8,
            pady=7,
            font=("Yu Gothic UI", 10),
        )
        self.prompt_text.grid(row=row + 1, column=0, sticky="ew")
        self._set_prompt_text(self._initial_prompt)
        row += 2

        self.osc_value_vars: dict[str, tk.StringVar] = {}
        for group_name, addresses in OSC_MONITOR_GROUPS:
            ttk.Label(
                content,
                text=group_name,
                style="Section.Panel.TLabel",
            ).grid(row=row, column=0, sticky="w", pady=(14, 4))
            row += 1
            for address in addresses:
                item = ttk.Frame(content, style="PanelAlt.TFrame", padding=(8, 5))
                item.grid(row=row, column=0, sticky="ew", pady=1)
                item.columnconfigure(0, weight=1)
                variable = tk.StringVar(value="-")
                self.osc_value_vars[address] = variable
                ttk.Label(
                    item,
                    text=address,
                    style="MetricName.TLabel",
                    font=("Consolas", 8),
                ).grid(row=0, column=0, sticky="w")
                ttk.Label(
                    item,
                    textvariable=variable,
                    style="Panel.TLabel",
                    wraplength=350,
                    justify="left",
                ).grid(row=1, column=0, sticky="w", pady=(2, 0))
                row += 1
        self._osc_receive_count = 0
        self._refresh_effective_settings_display()

    def _set_prompt_text(self, prompt: str) -> None:
        previous_state = str(self.prompt_text.cget("state"))
        if previous_state == "disabled":
            self.prompt_text.configure(state="normal")
        self.prompt_text.delete("1.0", "end")
        self.prompt_text.insert("1.0", prompt)
        if previous_state == "disabled":
            self.prompt_text.configure(state="disabled")

    def _refresh_effective_settings_display(self) -> None:
        variables = getattr(self, "effective_setting_vars", None)
        if not variables:
            return
        variables["model"].set(self.model_profile_var.get())
        variables["backend"].set(self.acceleration_backend_var.get())
        variables["performance"].set(self.performance_preset_var.get())
        variables["resolution"].set(self.resolution_var.get())
        variables["spout"].set(
            f"{self.spout_input_var.get()}  →  {self.spout_output_var.get()}  "
            f"({float(self.spout_sample_fps_var.get()):g} fps sample)"
        )
        strength = round((49 - int(round(self.denoise_var.get()))) / 49 * 100)
        variables["generation"].set(
            f"strength {strength}% / seed {int(self.seed_var.get())} / "
            f"target {float(self.target_fps_var.get()):g} fps"
        )
        variables["temporal"].set(
            f"input {float(self.temporal_feedback_var.get()):.2f} / "
            f"smooth {float(self.temporal_smoothing_var.get()):.2f} / "
            f"morph {float(self.latent_morph_strength_var.get()):.2f} / "
            f"history {int(round(self.latent_history_frames_var.get()))} / "
            f"motion {float(self.scene_cut_threshold_var.get()):.2f}"
        )

    def _build_ui(self) -> None:
        header = ttk.Frame(self, padding=(24, 18, 24, 12))
        header.pack(fill="x")
        ttk.Label(header, text="LOCAL DIFFUSION BRIDGE", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="TouchDesigner / Spout / StreamDiffusion — 完全ローカル画像生成",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(3, 0))

        body = ttk.Frame(self, padding=(20, 0, 20, 18))
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=0, minsize=390)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        settings_container = ttk.Frame(body, style="Panel.TFrame")
        settings_container.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        settings_container.rowconfigure(0, weight=1)
        settings_container.columnconfigure(0, weight=1)
        settings_canvas = tk.Canvas(
            settings_container,
            bg=PANEL,
            highlightthickness=0,
            width=370,
        )
        settings_canvas.grid(row=0, column=0, sticky="nsew")
        settings_scrollbar = ttk.Scrollbar(
            settings_container,
            orient="vertical",
            command=settings_canvas.yview,
        )
        settings_scrollbar.grid(row=0, column=1, sticky="ns")
        settings_canvas.configure(yscrollcommand=settings_scrollbar.set)
        settings = ttk.Frame(settings_canvas, style="Panel.TFrame", padding=18)
        settings_window = settings_canvas.create_window((0, 0), window=settings, anchor="nw")
        settings.bind(
            "<Configure>",
            lambda _event: settings_canvas.configure(
                scrollregion=settings_canvas.bbox("all")
            ),
        )
        settings_canvas.bind(
            "<Configure>",
            lambda event: settings_canvas.itemconfigure(settings_window, width=event.width),
        )
        scroll_settings = lambda event: settings_canvas.yview_scroll(
            -1 if event.delta > 0 else 1,
            "units",
        )
        settings_canvas.bind(
            "<Enter>",
            lambda _event: self.bind_all("<MouseWheel>", scroll_settings),
        )
        settings_canvas.bind(
            "<Leave>",
            lambda _event: self.unbind_all("<MouseWheel>"),
        )
        monitor = ttk.Frame(body, style="Panel.TFrame", padding=18)
        monitor.grid(row=0, column=1, sticky="nsew")
        monitor.columnconfigure(0, weight=1)
        monitor.rowconfigure(3, weight=1)

        self._build_settings(settings)
        self._build_monitor(monitor)

    def _build_settings(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        ttk.Label(parent, text="生成設定", style="Section.Panel.TLabel").grid(row=0, column=0, sticky="w")

        row = 1
        ttk.Label(parent, text="モデル選択", style="Muted.Panel.TLabel").grid(
            row=row, column=0, sticky="w", pady=(10, 3)
        )
        model_row = ttk.Frame(parent, style="Panel.TFrame")
        model_row.grid(row=row + 1, column=0, sticky="ew")
        model_row.columnconfigure(0, weight=1)
        self.model_profile_combo = ttk.Combobox(
            model_row,
            textvariable=self.model_profile_var,
            values=tuple(self._profile_by_display),
            state="readonly",
        )
        self.model_profile_combo.grid(row=0, column=0, sticky="ew")
        self.model_profile_combo.bind("<<ComboboxSelected>>", self._apply_model_profile)
        ttk.Button(model_row, text="再検索", command=self._refresh_model_profiles).grid(
            row=0, column=1, padx=(6, 0)
        )
        ttk.Label(
            parent,
            textvariable=self.model_description_var,
            style="Muted.Panel.TLabel",
            wraplength=335,
        ).grid(row=row + 2, column=0, sticky="w", pady=(3, 0))
        self.restart_controls = [self.model_profile_combo]
        row += 3

        row = self._labeled_entry(parent, row, "モデルパス（カスタム可）", self.model_path_var)
        row = self._labeled_entry(parent, row, "Spout 入力", self.spout_input_var)
        row = self._labeled_entry(parent, row, "Spout 出力", self.spout_output_var)

        ttk.Label(parent, text="推論バックエンド", style="Muted.Panel.TLabel").grid(
            row=row, column=0, sticky="w", pady=(10, 3)
        )
        backend = ttk.Combobox(
            parent,
            textvariable=self.acceleration_backend_var,
            values=tuple(BACKEND_LABELS),
            state="readonly",
        )
        backend.grid(row=row + 1, column=0, sticky="ew")
        backend.bind("<<ComboboxSelected>>", lambda _event: self._refresh_tensorrt_status())
        ttk.Label(
            parent,
            textvariable=self.tensorrt_status_var,
            style="Muted.Panel.TLabel",
            wraplength=335,
        ).grid(row=row + 2, column=0, sticky="w", pady=(3, 0))
        trt_buttons = ttk.Frame(parent, style="Panel.TFrame")
        trt_buttons.grid(row=row + 3, column=0, sticky="ew", pady=(5, 0))
        trt_buttons.columnconfigure(0, weight=1)
        trt_buttons.columnconfigure(1, weight=1)
        ttk.Button(
            trt_buttons,
            text="状態を更新",
            command=self._refresh_tensorrt_status,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 3))
        self.build_tensorrt_button = ttk.Button(
            trt_buttons,
            text="現在設定をビルド",
            command=self._build_tensorrt_engine,
        )
        self.build_tensorrt_button.grid(row=0, column=1, sticky="ew", padx=(3, 0))
        self.restart_controls.append(backend)
        row += 4

        ttk.Label(parent, text="動作プリセット", style="Muted.Panel.TLabel").grid(
            row=row, column=0, sticky="w", pady=(10, 3)
        )
        performance = ttk.Combobox(
            parent,
            textvariable=self.performance_preset_var,
            values=tuple(preset.label for preset in PERFORMANCE_PRESETS),
            state="readonly",
        )
        performance.grid(row=row + 1, column=0, sticky="ew")
        performance.bind("<<ComboboxSelected>>", self._apply_performance_preset)
        ttk.Label(
            parent,
            textvariable=self.performance_description_var,
            style="Muted.Panel.TLabel",
            wraplength=335,
        ).grid(row=row + 2, column=0, sticky="w", pady=(3, 0))
        self.restart_controls.append(performance)
        row += 3

        ttk.Label(parent, text="生成解像度", style="Muted.Panel.TLabel").grid(row=row, column=0, sticky="w", pady=(10, 3))
        resolution = ttk.Combobox(
            parent,
            textvariable=self.resolution_var,
            values=(
                "384 × 384",
                "448 × 448",
                "512 × 512",
                "640 × 640",
                "768 × 768",
                "896 × 896",
                "1024 × 1024",
            ),
            state="readonly",
        )
        resolution.grid(row=row + 1, column=0, sticky="ew")
        resolution.bind("<<ComboboxSelected>>", lambda _event: self._refresh_tensorrt_status())
        self.restart_controls.append(resolution)
        row += 2

        ttk.Label(parent, text="推論ステップ", style="Muted.Panel.TLabel").grid(
            row=row, column=0, sticky="w", pady=(10, 3)
        )
        lcm_steps = ttk.Combobox(
            parent,
            textvariable=self.lcm_steps_var,
            values=("1 step（高速）", "2 steps（高品質）"),
            state="readonly",
        )
        lcm_steps.grid(row=row + 1, column=0, sticky="ew")
        lcm_steps.bind("<<ComboboxSelected>>", lambda _event: self._refresh_tensorrt_status())
        self.restart_controls.append(lcm_steps)
        row += 2

        ttk.Label(parent, text="プロンプト", style="Muted.Panel.TLabel").grid(row=row, column=0, sticky="w", pady=(10, 3))
        self.prompt_text = tk.Text(
            parent,
            height=4,
            wrap="word",
            bg="#0e1218",
            fg=TEXT,
            insertbackground=TEXT,
            selectbackground="#2b6c55",
            relief="flat",
            padx=8,
            pady=7,
            font=("Yu Gothic UI", 10),
        )
        self.prompt_text.insert("1.0", self._initial_prompt)
        self.prompt_text.grid(row=row + 1, column=0, sticky="ew")
        row += 2

        self.update_prompt_button = ttk.Button(parent, text="プロンプトを即時更新", command=self._update_prompt)
        self.update_prompt_button.grid(row=row, column=0, sticky="ew", pady=(7, 2))
        row += 1

        ttk.Label(parent, textvariable=self.denoise_label_var, style="Muted.Panel.TLabel").grid(row=row, column=0, sticky="w", pady=(10, 0))
        denoise = ttk.Scale(parent, from_=0, to=49, variable=self.denoise_var, orient="horizontal")
        denoise.grid(row=row + 1, column=0, sticky="ew")
        row += 2

        row = self._labeled_scale(
            parent,
            row,
            self.temporal_feedback_label_var,
            self.temporal_feedback_var,
            0.0,
            0.8,
        )
        row = self._labeled_scale(
            parent,
            row,
            self.temporal_smoothing_label_var,
            self.temporal_smoothing_var,
            0.0,
            1.0,
        )
        row = self._labeled_scale(
            parent,
            row,
            self.latent_morph_label_var,
            self.latent_morph_strength_var,
            0.0,
            1.0,
        )
        row = self._labeled_scale(
            parent,
            row,
            self.latent_history_label_var,
            self.latent_history_frames_var,
            2,
            8,
        )
        row = self._labeled_scale(
            parent,
            row,
            self.scene_cut_label_var,
            self.scene_cut_threshold_var,
            0.0,
            1.0,
        )

        ttk.Button(
            parent,
            text="残像を抑えるモーフ推奨値",
            command=self._apply_morph_preset,
        ).grid(row=row, column=0, sticky="ew", pady=(7, 2))
        row += 1

        numeric = ttk.Frame(parent, style="Panel.TFrame")
        numeric.grid(row=row, column=0, sticky="ew", pady=(8, 0))
        for column in (0, 1, 2):
            numeric.columnconfigure(column, weight=1)
        ttk.Label(numeric, text="Seed", style="Muted.Panel.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(numeric, text="FPS 上限", style="Muted.Panel.TLabel").grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Label(numeric, text="Spout取得", style="Muted.Panel.TLabel").grid(row=0, column=2, sticky="w", padx=(8, 0))
        seed_entry = ttk.Spinbox(numeric, from_=-1, to=2147483647, textvariable=self.seed_var)
        seed_entry.grid(row=1, column=0, sticky="ew")
        fps_entry = ttk.Spinbox(numeric, from_=0.1, to=240, increment=1, textvariable=self.target_fps_var)
        fps_entry.grid(row=1, column=1, sticky="ew", padx=(8, 0))
        self.spout_fps_entry = ttk.Spinbox(
            numeric,
            from_=1,
            to=240,
            increment=1,
            textvariable=self.spout_sample_fps_var,
        )
        self.spout_fps_entry.grid(row=1, column=2, sticky="ew", padx=(8, 0))
        row += 1

        checks = ttk.Frame(parent, style="Panel.TFrame")
        checks.grid(row=row, column=0, sticky="ew", pady=(12, 0))
        ttk.Checkbutton(checks, text="LCM-LoRA", variable=self.use_lcm_lora_var).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(checks, text="TinyVAE（高速）", variable=self.tiny_vae_var).grid(row=0, column=1, sticky="w", padx=(12, 0))
        ttk.Checkbutton(checks, text="オフライン固定", variable=self.offline_var).grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(checks, text="入力を上下反転", variable=self.flip_input_var).grid(row=2, column=0, sticky="w")
        ttk.Checkbutton(checks, text="出力を上下反転", variable=self.flip_output_var).grid(row=2, column=1, sticky="w", padx=(12, 0))
        row += 1

        self.reset_temporal_button = ttk.Button(
            parent,
            text="時間履歴をリセット",
            command=self._reset_temporal,
        )
        self.reset_temporal_button.grid(row=row, column=0, sticky="ew", pady=(8, 0))
        row += 1

        ttk.Label(
            parent,
            text=(
                "※ 即時反映: プロンプト、変換強度、Seed、FPS、入力保持、"
                "出力平滑化、生成特徴モーフ/履歴、動き追従しきい値\n"
                "※ 再開が必要: モデル、バックエンド、Spout名/取得FPS、解像度、"
                "ステップ、LCM/TinyVAE、反転"
            ),
            style="Muted.Panel.TLabel",
            wraplength=335,
        ).grid(row=row, column=0, sticky="w", pady=(8, 0))
        row += 1

        buttons = ttk.Frame(parent, style="Panel.TFrame")
        buttons.grid(row=row, column=0, sticky="ew", pady=(16, 0))
        for column in (0, 1):
            buttons.columnconfigure(column, weight=1)
        self.start_button = ttk.Button(buttons, text="生成を開始", style="Accent.TButton", command=self._start)
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.stop_button = ttk.Button(buttons, text="停止", command=self._stop)
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        ttk.Button(parent, text="設定を保存", command=self._save_config).grid(row=row + 1, column=0, sticky="ew", pady=(8, 0))

    def _build_monitor(self, parent: ttk.Frame) -> None:
        status = ttk.Frame(parent, style="PanelAlt.TFrame", padding=12)
        status.grid(row=0, column=0, sticky="ew")
        status.columnconfigure(1, weight=1)
        self.status_dot = tk.Canvas(status, width=14, height=14, bg=PANEL_ALT, highlightthickness=0)
        self.status_dot.grid(row=0, column=0, rowspan=2, padx=(0, 10))
        self._status_oval = self.status_dot.create_oval(2, 2, 12, 12, fill=MUTED, outline="")
        ttk.Label(status, textvariable=self.status_var, style="MetricValue.TLabel").grid(row=0, column=1, sticky="w")
        ttk.Label(status, textvariable=self.status_detail_var, style="MetricName.TLabel").grid(row=1, column=1, sticky="w")

        metrics_frame = ttk.Frame(parent, style="Panel.TFrame")
        metrics_frame.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        for column in range(4):
            metrics_frame.columnconfigure(column, weight=1)
        metric_specs = (
            ("input_fps", "INPUT FPS"),
            ("source_fps", "SOURCE FPS"),
            ("output_fps", "OUTPUT FPS"),
            ("inference_fps", "INFER FPS"),
            ("inference_ms", "INFERENCE"),
            ("end_to_end_ms", "END TO END"),
            ("active_backend", "BACKEND"),
            ("gpu_utilization", "GPU LOAD"),
            ("vram", "VRAM"),
            ("gpu_temperature", "GPU TEMP"),
            ("input_resolution", "INPUT SIZE"),
            ("output_resolution", "OUTPUT SIZE"),
            ("spout_receive_ms", "SPOUT IN"),
            ("spout_send_ms", "SPOUT OUT"),
            ("input_age_ms", "INPUT AGE"),
            ("preprocess_ms", "PREPROCESS"),
            ("vae_encode_ms", "VAE ENCODE"),
            ("unet_ms", "UNET"),
            ("vae_decode_ms", "VAE DECODE"),
            ("postprocess_ms", "POSTPROCESS"),
            ("motion_score", "MOTION"),
            ("temporal_feedback", "INPUT HOLD"),
            ("latent_morph", "LATENT MORPH"),
        )
        self.metric_vars: dict[str, tk.StringVar] = {}
        for index, (key, label) in enumerate(metric_specs):
            card = ttk.Frame(metrics_frame, style="PanelAlt.TFrame", padding=(10, 8))
            card.grid(row=index // 4, column=index % 4, sticky="nsew", padx=3, pady=3)
            ttk.Label(card, text=label, style="MetricName.TLabel").pack(anchor="w")
            variable = tk.StringVar(value="-")
            self.metric_vars[key] = variable
            ttk.Label(card, textvariable=variable, style="MetricValue.TLabel").pack(anchor="w", pady=(2, 0))

        previews = ttk.Frame(parent, style="Panel.TFrame")
        previews.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        for column in (0, 1):
            previews.columnconfigure(column, weight=1)
        self.input_preview = self._preview_panel(previews, 0, "SPOUT INPUT")
        self.output_preview = self._preview_panel(previews, 1, "DIFFUSION OUTPUT")

        logs = ttk.Frame(parent, style="Panel.TFrame")
        logs.grid(row=3, column=0, sticky="nsew", pady=(12, 0))
        logs.columnconfigure(0, weight=1)
        logs.rowconfigure(1, weight=1)
        ttk.Label(logs, text="ログ", style="Section.Panel.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 5))
        self.log_text = tk.Text(
            logs,
            height=8,
            wrap="word",
            state="disabled",
            bg="#0b0f14",
            fg="#c8d4df",
            insertbackground=TEXT,
            relief="flat",
            padx=9,
            pady=7,
            font=("Consolas", 9),
        )
        self.log_text.grid(row=1, column=0, sticky="nsew")
        self._append_log("UIを起動しました。")

    def _labeled_entry(self, parent: ttk.Frame, row: int, label: str, variable: tk.Variable) -> int:
        ttk.Label(parent, text=label, style="Muted.Panel.TLabel").grid(row=row, column=0, sticky="w", pady=(10, 3))
        ttk.Entry(parent, textvariable=variable).grid(row=row + 1, column=0, sticky="ew")
        return row + 2

    def _labeled_scale(
        self,
        parent: ttk.Frame,
        row: int,
        label_variable: tk.StringVar,
        variable: tk.Variable,
        minimum: float,
        maximum: float,
    ) -> int:
        ttk.Label(parent, textvariable=label_variable, style="Muted.Panel.TLabel").grid(
            row=row,
            column=0,
            sticky="w",
            pady=(8, 0),
        )
        ttk.Scale(
            parent,
            from_=minimum,
            to=maximum,
            variable=variable,
            orient="horizontal",
        ).grid(row=row + 1, column=0, sticky="ew")
        return row + 2

    def _preview_panel(self, parent: ttk.Frame, column: int, title: str) -> tk.Label:
        panel = ttk.Frame(parent, style="PanelAlt.TFrame", padding=8)
        panel.grid(row=0, column=column, sticky="nsew", padx=(0, 4) if column == 0 else (4, 0))
        ttk.Label(panel, text=title, style="MetricName.TLabel").pack(anchor="w", pady=(0, 5))
        label = tk.Label(
            panel,
            text="NO SIGNAL",
            bg="#090c10",
            fg="#566273",
            width=34,
            height=13,
            font=("Consolas", 10),
        )
        label.pack(fill="both", expand=True)
        return label

    def _selected_model_profile(self) -> ModelProfile | None:
        return self._profile_by_display.get(self.model_profile_var.get())

    def _apply_model_profile(self, _event: object | None = None) -> None:
        profile = self._selected_model_profile()
        if profile is None:
            return
        self.model_path_var.set(profile.model_path)
        self.use_lcm_lora_var.set(profile.use_lcm_lora)
        description = profile.description
        if not profile.installed:
            description += f"  導入: .\\download_models.ps1 --preset {profile.key}"
        self.model_description_var.set(description)

        preset = find_performance_preset(profile.recommended_preset)
        if preset is not None:
            self.performance_preset_var.set(preset.label)
            self._apply_performance_preset()
        self._refresh_tensorrt_status()

    def _apply_performance_preset(self, _event: object | None = None) -> None:
        preset = next(
            (
                candidate
                for candidate in PERFORMANCE_PRESETS
                if candidate.label == self.performance_preset_var.get()
            ),
            None,
        )
        if preset is None:
            return
        self.lcm_steps_var.set(
            "1 step（高速）" if preset.steps == 1 else "2 steps（高品質）"
        )
        self.tiny_vae_var.set(preset.use_tiny_vae)
        self.performance_description_var.set(preset.description)
        self._refresh_tensorrt_status()

    def _refresh_model_profiles(self) -> None:
        previous = self._selected_model_profile()
        previous_key = previous.key if previous else ""
        self.model_profiles = discover_model_profiles()
        self._profile_by_display = {
            profile.display_label: profile for profile in self.model_profiles
        }
        self.model_profile_combo.configure(values=tuple(self._profile_by_display))
        profile = find_profile(
            self.model_profiles,
            previous_key,
            self.model_path_var.get().strip(),
        )
        if profile is not None:
            self.model_profile_var.set(profile.display_label)
            self.model_description_var.set(profile.description)
        self._append_log("ローカルモデルを再検索しました。")

    def _refresh_tensorrt_status(self) -> None:
        try:
            config = self._collect_config()
            status = inspect_tensorrt(config)
            self.tensorrt_status_var.set(status.message)
        except (KeyError, tk.TclError, TypeError, ValueError) as exc:
            self.tensorrt_status_var.set(f"設定を確認してください: {exc}")

    def _build_tensorrt_engine(self) -> None:
        if self.worker.running or self._trt_build_process is not None:
            return
        project_dir = Path(__file__).resolve().parents[1]
        trt_python = project_dir / ".venv-trt" / "Scripts" / "python.exe"
        if not trt_python.is_file():
            messagebox.showerror(
                "TensorRT環境がありません",
                "先にPowerShellで .\\setup_tensorrt.ps1 を実行してください。",
            )
            return
        try:
            config = self._collect_config()
            config.validate()
        except Exception as exc:
            messagebox.showerror("ビルドできません", str(exc))
            return
        if not messagebox.askyesno(
            "TensorRTエンジンを作成",
            "現在のモデル・解像度・ステップ・GPU専用のFP16エンジンを作成します。\n"
            "初回は10〜30分程度かかり、ビルド中はGPUを使用します。続行しますか？",
        ):
            return
        config.save(self.config_path)
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        command = [
            str(trt_python),
            str(project_dir / "build_tensorrt_engine.py"),
            "--config",
            str(self.config_path),
        ]
        try:
            self._trt_build_process = subprocess.Popen(
                command,
                cwd=project_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=environment,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            self._trt_build_process = None
            messagebox.showerror("ビルドを開始できません", str(exc))
            return
        self.build_tensorrt_button.configure(state="disabled")
        self.tensorrt_status_var.set("TensorRTエンジンをビルド中…")
        self._append_log("TensorRTエンジンのビルドを開始しました。")
        threading.Thread(target=self._read_tensorrt_build, daemon=True).start()

    def _read_tensorrt_build(self) -> None:
        process = self._trt_build_process
        if process is None:
            return
        if process.stdout is not None:
            for line in process.stdout:
                self.events.put(WorkerEvent("log", line.rstrip()))
        return_code = process.wait()
        self.events.put(WorkerEvent("tensorrt_build_complete", return_code))

    def _collect_config(self) -> AppConfig:
        resolution = self.resolution_var.get().replace(" ", "").split("×")
        if len(resolution) != 2:
            raise ValueError("生成解像度の形式が正しくありません。")
        profile = self._selected_model_profile()
        model_path = self.model_path_var.get().strip()
        profile_key = "custom"
        if (
            profile is not None
            and profile.model_path.replace("\\", "/").lower()
            == model_path.replace("\\", "/").lower()
        ):
            profile_key = profile.key
        return AppConfig(
            model_profile=profile_key,
            model_path=model_path,
            lcm_lora_path=self.lcm_lora_path_var.get().strip(),
            tiny_vae_path="models/taesd",
            spout_input=self.spout_input_var.get().strip(),
            spout_output=self.spout_output_var.get().strip(),
            prompt=self.prompt_text.get("1.0", "end").strip(),
            width=int(resolution[0]),
            height=int(resolution[1]),
            denoise_index=int(round(self.denoise_var.get())),
            seed=int(self.seed_var.get()),
            target_fps=float(self.target_fps_var.get()),
            spout_sample_fps=float(self.spout_sample_fps_var.get()),
            acceleration_backend=BACKEND_LABELS[self.acceleration_backend_var.get()],
            tensorrt_engine_root="engines/tensorrt",
            tensorrt_cuda_graph=False,
            use_lcm_lora=bool(self.use_lcm_lora_var.get()),
            lcm_steps=int(self.lcm_steps_var.get().split()[0]),
            use_tiny_vae=bool(self.tiny_vae_var.get()),
            temporal_feedback=float(self.temporal_feedback_var.get()),
            temporal_smoothing=float(self.temporal_smoothing_var.get()),
            latent_morph_strength=float(self.latent_morph_strength_var.get()),
            latent_history_frames=int(round(self.latent_history_frames_var.get())),
            scene_cut_threshold=float(self.scene_cut_threshold_var.get()),
            flip_input=bool(self.flip_input_var.get()),
            flip_output=bool(self.flip_output_var.get()),
            offline_mode=bool(self.offline_var.get()),
        )

    def _start(self) -> None:
        try:
            config = self._collect_config()
            config.validate()
            config.save(self.config_path)
            self.worker.start(config)
            self._generation_desired = True
            self._append_log("生成開始を要求しました。")
            self._set_controls_for_state("起動中")
        except Exception as exc:
            messagebox.showerror("開始できません", str(exc))

    def _stop(self) -> None:
        self._generation_desired = False
        self._cancel_osc_restart()
        self.worker.stop()
        self.status_detail_var.set("現在の推論が終了するまで待っています…")
        self._append_log("停止を要求しました。")

    def _update_prompt(self) -> None:
        prompt = self.prompt_text.get("1.0", "end").strip()
        if not prompt:
            messagebox.showwarning("プロンプト", "プロンプトを入力してください。")
            return
        if not self.worker.running:
            self._append_log("プロンプトを設定しました。次回開始時に反映されます。")
            return
        self.worker.update_prompt(prompt)

    def _on_denoise_changed(self, *_args: object) -> None:
        self._update_denoise_label()
        self._schedule_live_settings_update()

    def _on_temporal_changed(self, *_args: object) -> None:
        self._update_temporal_labels()
        self._schedule_live_settings_update()

    def _on_live_value_changed(self, *_args: object) -> None:
        self._schedule_live_settings_update()

    def _apply_morph_preset(self) -> None:
        """Favor history-clamped latent motion over recursive RGB feedback."""
        self.temporal_feedback_var.set(0.0)
        self.temporal_smoothing_var.set(0.5)
        self.latent_morph_strength_var.set(0.65)
        self.latent_history_frames_var.set(8)
        self.scene_cut_threshold_var.set(0.8)
        self._append_log("モーフ推奨値を設定しました（実行中も即時反映）。")

    def _schedule_live_settings_update(self) -> None:
        if self._state not in {"起動中", "実行中"} or not self.worker.running:
            return
        if self._live_update_after_id is not None:
            self.after_cancel(self._live_update_after_id)
        # Coalesce slider movement and partially typed Spinbox values. This keeps
        # temporal controls responsive and avoids repeatedly preparing timesteps.
        self._live_update_after_id = self.after(180, self._apply_live_settings)

    def _apply_live_settings(self) -> None:
        self._live_update_after_id = None
        if not self.worker.running:
            return
        try:
            settings = {
                "denoise_index": int(round(self.denoise_var.get())),
                "seed": int(self.seed_var.get()),
                "target_fps": float(self.target_fps_var.get()),
                "temporal_feedback": float(self.temporal_feedback_var.get()),
                "temporal_smoothing": float(self.temporal_smoothing_var.get()),
                "latent_morph_strength": float(self.latent_morph_strength_var.get()),
                "latent_history_frames": int(round(self.latent_history_frames_var.get())),
                "scene_cut_threshold": float(self.scene_cut_threshold_var.get()),
            }
            if not 0 <= settings["denoise_index"] <= 49:
                raise ValueError("変換強度が範囲外です。")
            if not 0.1 <= settings["target_fps"] <= 240:
                raise ValueError("FPS上限が範囲外です。")
            if not 0.0 <= settings["temporal_feedback"] <= 0.8:
                raise ValueError("入力フレーム保持が範囲外です。")
            if not 0.0 <= settings["temporal_smoothing"] <= 1.0:
                raise ValueError("出力平滑化が範囲外です。")
            if not 0.0 <= settings["latent_morph_strength"] <= 1.0:
                raise ValueError("生成特徴モーフが範囲外です。")
            if not 2 <= settings["latent_history_frames"] <= 8:
                raise ValueError("特徴履歴フレームが範囲外です。")
            if not 0.0 <= settings["scene_cut_threshold"] <= 1.0:
                raise ValueError("動き追従しきい値が範囲外です。")
        except (tk.TclError, TypeError, ValueError):
            # Spinbox text may be temporarily empty while the user is editing it.
            return
        self.worker.update_live_settings(settings)

    def _reset_temporal(self) -> None:
        if self.worker.running:
            self.worker.reset_temporal()
        else:
            self._append_log("時間履歴は次回開始時に初期化されます。")

    def _save_config(self) -> None:
        try:
            config = self._collect_config()
            config.save(self.config_path)
            self._append_log(f"設定を保存しました: {self.config_path}")
        except Exception as exc:
            messagebox.showerror("保存できません", str(exc))

    def _start_osc_server(self) -> None:
        server = OSCUDPServer(self.osc_events, OSC_HOST, OSC_PORT)
        try:
            server.start()
        except OSError as exc:
            self._append_log(f"OSC受信を開始できません: {exc}")
            self.status_detail_var.set(f"OSC UDP {OSC_PORT}を開けません。")
            if hasattr(self, "osc_listener_var"):
                self.osc_listener_var.set(f"受信エラー: {exc}")
            return
        self.osc_server = server
        if hasattr(self, "osc_listener_var"):
            self.osc_listener_var.set(f"受信中: udp://{OSC_HOST}:{server.port}")
        self._append_log(
            f"OSC受信を開始しました: udp://{OSC_HOST}:{server.port}"
        )

    def _poll_osc_events(self) -> None:
        # TouchDesigner OSC Out CHOP can publish every channel every frame.
        # Coalesce one UI tick to the newest packet per address so a 60 fps
        # control stream cannot monopolize Tk's event loop.
        latest_messages: dict[str, OSCMessage] = {}
        trigger_messages: list[OSCMessage] = []
        last_message: OSCMessage | None = None
        received = 0
        for _index in range(512):
            try:
                event = self.osc_events.get_nowait()
            except queue.Empty:
                break
            if event.kind == "error":
                self._append_log(str(event.data))
                continue
            if not isinstance(event.data, OSCMessage):
                continue
            if event.data.address in OSC_TRIGGER_ADDRESSES:
                # Preserve 1 -> 0 pulses even when both edges arrive within one
                # 80 ms UI poll. Ordinary parameters only need their latest value.
                trigger_messages.append(event.data)
            else:
                latest_messages[event.data.address] = event.data
            last_message = event.data
            received += 1

        if received:
            self._osc_receive_count += received
            self.osc_count_var.set(f"受信数: {self._osc_receive_count}")
        for message in (*latest_messages.values(), *trigger_messages):
            self._record_osc_value(message)
            if not self._osc_trigger_has_rising_edge(message):
                continue
            try:
                command = decode_control_message(message)
                if command is not None:
                    self._handle_osc_command(command)
            except (KeyError, TypeError, ValueError) as exc:
                self._append_log(f"OSC値を反映できません: {exc}")
        if last_message is not None:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            self.osc_last_var.set(
                f"最終受信 {timestamp}  {last_message.address} = "
                f"{self._format_osc_value(last_message)}"
            )

    @staticmethod
    def _format_osc_value(message: OSCMessage) -> str:
        if not message.args:
            return "trigger"
        formatted: list[str] = []
        for item in message.args:
            if isinstance(item, float):
                formatted.append(f"{item:.6g}")
            else:
                formatted.append(str(item))
        return ", ".join(formatted)

    def _record_osc_value(self, message: OSCMessage) -> None:
        variable = self.osc_value_vars.get(message.address)
        if variable is not None:
            value = self._format_osc_value(message)
            if variable.get() != value:
                variable.set(value)

    def _osc_trigger_has_rising_edge(self, message: OSCMessage) -> bool:
        if message.address not in OSC_TRIGGER_ADDRESSES:
            return True
        if not message.args:
            active = True
        else:
            value = message.args[0]
            active = isinstance(value, bool) and value or (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and float(value) > 0.0
            )
        was_active = self._osc_trigger_active.get(message.address, False)
        self._osc_trigger_active[message.address] = active
        if not active or was_active:
            return False
        # The launcher sends only positive triggers. Let the opposite transport
        # command re-arm Start/Stop even when no explicit falling edge is sent.
        if message.address == "/ergonomics/system/start":
            self._osc_trigger_active["/ergonomics/system/stop"] = False
        elif message.address == "/ergonomics/system/stop":
            self._osc_trigger_active["/ergonomics/system/start"] = False
        return True

    def _handle_osc_command(self, command: OSCCommand) -> None:
        if command.category == "live":
            if self._osc_last_live_values.get(command.name) == command.value:
                return
            self._osc_last_live_values[command.name] = command.value
            self._apply_osc_live_value(command.name, command.value)
            return
        if command.category == "config":
            if self._osc_last_config_values.get(command.name) == command.value:
                return
            self._osc_last_config_values[command.name] = command.value
            self._osc_pending_config[command.name] = command.value
            self._schedule_osc_config_apply()
            return
        if command.name == "apply":
            self._cancel_osc_config_timer()
            self._apply_pending_osc_config()
        elif command.name == "start":
            self._start_from_osc()
        elif command.name == "stop":
            self._stop_from_osc()
        elif command.name == "reset_temporal":
            if self.worker.running:
                self.worker.reset_temporal()
                self._append_log("OSCから時間安定化の履歴をリセットしました。")
            else:
                self._append_log("OSC Resetを受信しました（履歴はすでに空です）。")

    def _apply_osc_live_value(self, name: str, value: Any) -> None:
        if name == "prompt":
            prompt = str(value)
            if self.prompt_text.get("1.0", "end").strip() != prompt:
                self._set_prompt_text(prompt)
            if self.worker.running:
                self.worker.update_prompt(prompt)
            self._refresh_effective_settings_display()
            return

        variables = {
            "denoise_index": self.denoise_var,
            "seed": self.seed_var,
            "target_fps": self.target_fps_var,
            "temporal_feedback": self.temporal_feedback_var,
            "temporal_smoothing": self.temporal_smoothing_var,
            "latent_morph_strength": self.latent_morph_strength_var,
            "latent_history_frames": self.latent_history_frames_var,
            "scene_cut_threshold": self.scene_cut_threshold_var,
        }
        variable = variables.get(name)
        if variable is None:
            raise ValueError(f"未対応のライブ設定です: {name}")
        variable.set(value)
        self._refresh_effective_settings_display()
        if self.worker.running:
            # Send this value immediately. The existing UI trace still coalesces
            # rapid slider movement and is harmless if it later repeats the state.
            self.worker.update_live_settings({name: value})

    def _schedule_osc_config_apply(self) -> None:
        self._cancel_osc_config_timer()
        self._osc_config_after_id = self.after(
            OSC_CONFIG_DEBOUNCE_MS,
            self._apply_pending_osc_config,
        )

    def _cancel_osc_config_timer(self) -> None:
        if self._osc_config_after_id is None:
            return
        try:
            self.after_cancel(self._osc_config_after_id)
        except tk.TclError:
            pass
        self._osc_config_after_id = None

    def _apply_pending_osc_config(self, *, start_if_stopped: bool = False) -> None:
        self._cancel_osc_config_timer()
        if not self._osc_pending_config:
            if start_if_stopped:
                self._launch_current_config_from_osc()
            return

        updates = self._osc_pending_config
        self._osc_pending_config = {}
        try:
            config = apply_osc_config(self._collect_config(), updates)
            self._sync_ui_from_config(config)
            config.save(self.config_path)
        except Exception as exc:
            self._append_log(f"OSC設定を適用できません: {exc}")
            self.status_detail_var.set(str(exc).splitlines()[0])
            if start_if_stopped:
                self._generation_desired = False
            return

        names = ", ".join(sorted(updates))
        self._append_log(f"OSC設定を適用しました: {names}")
        if self._generation_desired and (
            self.worker.running or self._osc_restart_config is not None
        ):
            self._restart_worker_for_osc(config)
        elif start_if_stopped:
            self._launch_worker_from_osc(config)

    def _sync_ui_from_config(self, config: AppConfig) -> None:
        profile = find_profile(self.model_profiles, config.model_profile, config.model_path)
        if profile is not None:
            self.model_profile_var.set(profile.display_label)
            description = profile.description
            if not profile.installed:
                description += f"  導入: .\\download_models.ps1 --preset {profile.key}"
            self.model_description_var.set(description)
        self.model_path_var.set(config.model_path)
        self.lcm_lora_path_var.set(config.lcm_lora_path)
        self.spout_input_var.set(config.spout_input)
        self.spout_output_var.set(config.spout_output)
        self.resolution_var.set(f"{config.width} × {config.height}")
        self.denoise_var.set(config.denoise_index)
        self.seed_var.set(config.seed)
        self.target_fps_var.set(config.target_fps)
        self.spout_sample_fps_var.set(config.spout_sample_fps)
        backend_label = next(
            label
            for label, backend in BACKEND_LABELS.items()
            if backend == config.acceleration_backend
        )
        self.acceleration_backend_var.set(backend_label)
        self.use_lcm_lora_var.set(config.use_lcm_lora)
        self.lcm_steps_var.set(
            "1 step（高速）" if config.lcm_steps == 1 else "2 steps（高品質）"
        )
        preset = infer_performance_preset(config.lcm_steps, config.use_tiny_vae)
        self.performance_preset_var.set(preset.label)
        self.performance_description_var.set(preset.description)
        self.tiny_vae_var.set(config.use_tiny_vae)
        self.temporal_feedback_var.set(config.temporal_feedback)
        self.temporal_smoothing_var.set(config.temporal_smoothing)
        self.latent_morph_strength_var.set(config.latent_morph_strength)
        self.latent_history_frames_var.set(config.latent_history_frames)
        self.scene_cut_threshold_var.set(config.scene_cut_threshold)
        self.flip_input_var.set(False)
        self.flip_output_var.set(False)
        self.offline_var.set(True)
        if self.prompt_text.get("1.0", "end").strip() != config.prompt:
            self._set_prompt_text(config.prompt)
        self._update_denoise_label()
        self._update_temporal_labels()
        self._refresh_effective_settings_display()
        self._refresh_tensorrt_status()

    def _start_from_osc(self) -> None:
        self._generation_desired = True
        self._cancel_osc_config_timer()
        if self._osc_pending_config:
            self._apply_pending_osc_config(start_if_stopped=True)
            return
        if self.worker.running or self._osc_restart_config is not None:
            self._append_log("OSC Startを受信しましたが、推論はすでに動作中です。")
            return
        self._launch_current_config_from_osc()

    def _launch_current_config_from_osc(self) -> None:
        try:
            config = self._collect_config()
            config.validate()
            config.save(self.config_path)
        except Exception as exc:
            self._append_log(f"OSC Startを実行できません: {exc}")
            self.status_detail_var.set(str(exc).splitlines()[0])
            self._generation_desired = False
            return
        self._launch_worker_from_osc(config)

    def _launch_worker_from_osc(self, config: AppConfig) -> None:
        try:
            self.worker.start(config)
        except Exception as exc:
            self._append_log(f"OSC Startを実行できません: {exc}")
            self.status_detail_var.set(str(exc).splitlines()[0])
            self._generation_desired = False
            return
        self._generation_desired = True
        self._append_log("OSCから生成開始を要求しました。")
        self._set_controls_for_state("起動中")

    def _restart_worker_for_osc(self, config: AppConfig) -> None:
        self._osc_restart_config = config
        if self.worker.running:
            self.worker.stop()
            self.status_detail_var.set("OSC設定変更のため推論を再初期化しています…")
            self._append_log("OSCの非リアルタイム設定を反映するため再起動します。")
        self._schedule_osc_restart_check()

    def _schedule_osc_restart_check(self) -> None:
        if self._osc_restart_after_id is not None:
            return
        self._osc_restart_after_id = self.after(100, self._check_osc_restart)

    def _check_osc_restart(self) -> None:
        self._osc_restart_after_id = None
        if (
            self._closing
            or not self._generation_desired
            or self._osc_restart_config is None
        ):
            return
        if self.worker.running:
            self._schedule_osc_restart_check()
            return
        config = self._osc_restart_config
        self._osc_restart_config = None
        self._launch_worker_from_osc(config)

    def _cancel_osc_restart(self) -> None:
        self._osc_restart_config = None
        if self._osc_restart_after_id is None:
            return
        try:
            self.after_cancel(self._osc_restart_after_id)
        except tk.TclError:
            pass
        self._osc_restart_after_id = None

    def _stop_from_osc(self) -> None:
        self._generation_desired = False
        self._cancel_osc_restart()
        if self.worker.running:
            self.worker.stop()
            self.status_detail_var.set("OSC Stop: 現在の推論終了を待っています…")
            self._append_log("OSCから停止を要求しました。")
        else:
            self._append_log("OSC Stopを受信しました（推論は停止済みです）。")

    def _poll_events(self) -> None:
        self._poll_osc_events()
        try:
            while True:
                event = self.events.get_nowait()
                self._handle_event(event)
        except queue.Empty:
            pass
        if not self._closing:
            self.after(80, self._poll_events)

    def _handle_event(self, event: WorkerEvent) -> None:
        if event.kind == "log":
            self._append_log(str(event.data))
        elif event.kind == "state":
            self._set_controls_for_state(str(event.data))
        elif event.kind == "metrics":
            self._update_metrics(event.data)
        elif event.kind == "preview":
            self._set_preview("input", self.input_preview, event.data["input"])
            self._set_preview("output", self.output_preview, event.data["output"])
        elif event.kind == "error":
            self.status_detail_var.set(str(event.data).splitlines()[0])
        elif event.kind == "tensorrt_build_complete":
            return_code = int(event.data)
            self._trt_build_process = None
            if hasattr(self, "build_tensorrt_button"):
                self.build_tensorrt_button.configure(state="normal")
            if return_code == 0:
                self._append_log("TensorRTエンジンのビルドが完了しました。")
            else:
                self._append_log(f"TensorRTビルドはエラーで終了しました (code={return_code})。")
            self._refresh_tensorrt_status()

    def _update_metrics(self, metrics: dict[str, Any]) -> None:
        self.status_detail_var.set(str(metrics.get("status", "")))
        self.metric_vars["input_fps"].set(f"{metrics.get('input_fps', 0):.1f}")
        source_fps = metrics.get("source_fps", 0)
        self.metric_vars["source_fps"].set(f"{source_fps:.1f}" if source_fps else "-")
        self.metric_vars["output_fps"].set(f"{metrics.get('output_fps', 0):.1f}")
        self.metric_vars["inference_fps"].set(f"{metrics.get('inference_fps', 0):.1f}")
        self.metric_vars["inference_ms"].set(f"{metrics.get('inference_ms', 0):.1f} ms")
        self.metric_vars["end_to_end_ms"].set(f"{metrics.get('end_to_end_ms', 0):.1f} ms")
        self.metric_vars["active_backend"].set(str(metrics.get("active_backend", "-")))
        self.metric_vars["gpu_utilization"].set(f"{metrics.get('gpu_utilization', 0):.0f} %")
        used = metrics.get("vram_used_mb", 0) / 1024
        total = metrics.get("vram_total_mb", 0) / 1024
        self.metric_vars["vram"].set(f"{used:.1f}/{total:.1f} GB" if total else "-")
        temp = metrics.get("gpu_temperature", 0)
        self.metric_vars["gpu_temperature"].set(f"{temp:.0f} °C" if temp else "-")
        self.metric_vars["input_resolution"].set(str(metrics.get("input_resolution", "-")))
        self.metric_vars["output_resolution"].set(str(metrics.get("output_resolution", "-")))
        for key in (
            "spout_receive_ms",
            "spout_send_ms",
            "input_age_ms",
            "preprocess_ms",
            "vae_encode_ms",
            "unet_ms",
            "vae_decode_ms",
            "postprocess_ms",
        ):
            self.metric_vars[key].set(f"{metrics.get(key, 0):.1f} ms")
        self.metric_vars["motion_score"].set(f"{metrics.get('motion_score', 0) * 100:.1f} %")
        self.metric_vars["temporal_feedback"].set(
            f"{metrics.get('temporal_feedback', 0) * 100:.0f} %"
        )
        self.metric_vars["latent_morph"].set(
            f"{metrics.get('latent_morph', 0) * 100:.0f} %"
        )

    def _set_preview(self, key: str, label: tk.Label, image: Image.Image) -> None:
        canvas = Image.new("RGB", (256, 256), "#090c10")
        x = (canvas.width - image.width) // 2
        y = (canvas.height - image.height) // 2
        canvas.paste(image, (x, y))
        photo = ImageTk.PhotoImage(canvas)
        self._preview_refs[key] = photo
        label.configure(image=photo, text="", width=256, height=256)

    def _set_controls_for_state(self, state: str) -> None:
        self._state = state
        self.status_var.set(state)
        color = {
            "停止": MUTED,
            "起動中": WARNING,
            "実行中": ACCENT,
            "エラー": ERROR,
        }.get(state, MUTED)
        self.status_dot.itemconfigure(self._status_oval, fill=color)
        is_running = state in {"起動中", "実行中"}
        if hasattr(self, "start_button"):
            self.start_button.configure(state="disabled" if is_running else "normal")
        if hasattr(self, "stop_button"):
            self.stop_button.configure(state="normal" if is_running else "disabled")
        if hasattr(self, "update_prompt_button"):
            self.update_prompt_button.configure(
                state="normal" if state == "実行中" else "disabled"
            )
        if hasattr(self, "reset_temporal_button"):
            self.reset_temporal_button.configure(
                state="normal" if state == "実行中" else "disabled"
            )
        for control in self.restart_controls:
            control.configure(state="disabled" if is_running else "readonly")
        if hasattr(self, "spout_fps_entry"):
            self.spout_fps_entry.configure(
                state="disabled" if is_running else "normal"
            )
        build_available = not is_running and self._trt_build_process is None
        if hasattr(self, "build_tensorrt_button"):
            self.build_tensorrt_button.configure(
                state="normal" if build_available else "disabled"
            )

    def _update_denoise_label(self) -> None:
        try:
            value = int(round(self.denoise_var.get()))
        except tk.TclError:
            return
        strength = round((49 - value) / 49 * 100)
        self.denoise_label_var.set(
            f"変換の強さ  {strength}%  （入力保持 {100 - strength}%・実行中も反映）"
        )

    def _update_temporal_labels(self) -> None:
        try:
            feedback = round(float(self.temporal_feedback_var.get()) * 100)
            smoothing = round(float(self.temporal_smoothing_var.get()) * 100)
            latent_morph = round(float(self.latent_morph_strength_var.get()) * 100)
            latent_frames = int(round(self.latent_history_frames_var.get()))
            scene_cut = round(float(self.scene_cut_threshold_var.get()) * 100)
        except tk.TclError:
            return
        self.temporal_feedback_label_var.set(f"入力フレーム保持  {feedback}%")
        self.temporal_smoothing_label_var.set(f"出力平滑化  {smoothing}%")
        self.latent_morph_label_var.set(f"生成特徴モーフ  {latent_morph}%")
        self.latent_history_label_var.set(f"特徴履歴  {latent_frames} フレーム")
        self.scene_cut_label_var.set(f"動き追従しきい値（自動リセットなし）  {scene_cut}%")

    def _append_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"[{timestamp}] {message.rstrip()}\n")
        line_count = int(self.log_text.index("end-1c").split(".")[0])
        if line_count > 500:
            self.log_text.delete("1.0", "100.0")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _on_close(self) -> None:
        self._closing = True
        self._generation_desired = False
        self._cancel_osc_config_timer()
        self._cancel_osc_restart()
        if self._live_update_after_id is not None:
            self.after_cancel(self._live_update_after_id)
            self._live_update_after_id = None
        if self.osc_server is not None:
            self.osc_server.close()
            self.osc_server = None
        self.worker.stop()
        self.destroy()


def run_ui(
    config_path: Path = DEFAULT_CONFIG_PATH,
    *,
    autostart: bool = False,
) -> None:
    app = DiffusionApp(config_path=config_path, autostart=autostart)
    app.mainloop()
