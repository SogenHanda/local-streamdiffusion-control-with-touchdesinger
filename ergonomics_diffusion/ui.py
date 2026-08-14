from __future__ import annotations

import queue
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
from .runtime import DiffusionWorker, WorkerEvent


BACKGROUND = "#11151c"
PANEL = "#1a202b"
PANEL_ALT = "#222a37"
TEXT = "#eef3f8"
MUTED = "#9aa8b8"
ACCENT = "#69d2a3"
WARNING = "#f2c14e"
ERROR = "#ef6a6a"


class DiffusionApp(tk.Tk):
    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH) -> None:
        super().__init__()
        self.title("Ergonomics Local Diffusion")
        self.geometry("1220x980")
        self.minsize(1060, 820)
        self.configure(bg=BACKGROUND)

        self.config_path = config_path
        self.events: "queue.Queue[WorkerEvent]" = queue.Queue(maxsize=16)
        self.worker = DiffusionWorker(self.events)
        self._preview_refs: dict[str, ImageTk.PhotoImage] = {}
        self._closing = False
        self._state = "停止"
        self._live_update_after_id: str | None = None

        try:
            config = AppConfig.load(config_path)
        except Exception as exc:
            config = AppConfig()
            messagebox.showwarning("設定読込エラー", f"既定設定で起動します。\n\n{exc}")

        self._configure_styles()
        self._create_variables(config)
        self._build_ui()
        self._set_controls_for_state("停止")
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(80, self._poll_events)

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
        self.scene_cut_threshold_var = tk.DoubleVar(value=config.scene_cut_threshold)
        self.temporal_feedback_label_var = tk.StringVar()
        self.temporal_smoothing_label_var = tk.StringVar()
        self.scene_cut_label_var = tk.StringVar()
        self.flip_input_var = tk.BooleanVar(value=config.flip_input)
        self.flip_output_var = tk.BooleanVar(value=config.flip_output)
        self.offline_var = tk.BooleanVar(value=config.offline_mode)
        self.status_var = tk.StringVar(value="停止")
        self.status_detail_var = tk.StringVar(value="設定を確認して開始してください。")
        self.denoise_var.trace_add("write", self._on_denoise_changed)
        self.seed_var.trace_add("write", self._on_live_value_changed)
        self.target_fps_var.trace_add("write", self._on_live_value_changed)
        self.temporal_feedback_var.trace_add("write", self._on_temporal_changed)
        self.temporal_smoothing_var.trace_add("write", self._on_temporal_changed)
        self.scene_cut_threshold_var.trace_add("write", self._on_temporal_changed)
        self._update_denoise_label()
        self._update_temporal_labels()
        self._initial_prompt = config.prompt

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
            0.5,
        )
        row = self._labeled_scale(
            parent,
            row,
            self.scene_cut_label_var,
            self.scene_cut_threshold_var,
            0.05,
            0.8,
        )

        numeric = ttk.Frame(parent, style="Panel.TFrame")
        numeric.grid(row=row, column=0, sticky="ew", pady=(8, 0))
        for column in (0, 1):
            numeric.columnconfigure(column, weight=1)
        ttk.Label(numeric, text="Seed", style="Muted.Panel.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(numeric, text="FPS 上限", style="Muted.Panel.TLabel").grid(row=0, column=1, sticky="w", padx=(8, 0))
        seed_entry = ttk.Spinbox(numeric, from_=-1, to=2147483647, textvariable=self.seed_var)
        seed_entry.grid(row=1, column=0, sticky="ew")
        fps_entry = ttk.Spinbox(numeric, from_=0.1, to=240, increment=1, textvariable=self.target_fps_var)
        fps_entry.grid(row=1, column=1, sticky="ew", padx=(8, 0))
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
                "出力平滑化、シーン変化リセット\n"
                "※ 再開が必要: モデル、Spout名、解像度、ステップ、LCM/TinyVAE、反転"
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
            ("output_fps", "OUTPUT FPS"),
            ("inference_fps", "INFER FPS"),
            ("inference_ms", "INFERENCE"),
            ("gpu_utilization", "GPU LOAD"),
            ("vram", "VRAM"),
            ("gpu_temperature", "GPU TEMP"),
            ("input_resolution", "INPUT SIZE"),
            ("output_resolution", "OUTPUT SIZE"),
            ("motion_score", "MOTION"),
            ("temporal_feedback", "INPUT HOLD"),
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
        variable: tk.DoubleVar,
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
            use_lcm_lora=bool(self.use_lcm_lora_var.get()),
            lcm_steps=int(self.lcm_steps_var.get().split()[0]),
            use_tiny_vae=bool(self.tiny_vae_var.get()),
            temporal_feedback=float(self.temporal_feedback_var.get()),
            temporal_smoothing=float(self.temporal_smoothing_var.get()),
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
            self._append_log("生成開始を要求しました。")
            self._set_controls_for_state("起動中")
        except Exception as exc:
            messagebox.showerror("開始できません", str(exc))

    def _stop(self) -> None:
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
                "scene_cut_threshold": float(self.scene_cut_threshold_var.get()),
            }
            if not 0 <= settings["denoise_index"] <= 49:
                raise ValueError("変換強度が範囲外です。")
            if not 0.1 <= settings["target_fps"] <= 240:
                raise ValueError("FPS上限が範囲外です。")
            if not 0.0 <= settings["temporal_feedback"] <= 0.8:
                raise ValueError("入力フレーム保持が範囲外です。")
            if not 0.0 <= settings["temporal_smoothing"] <= 0.8:
                raise ValueError("出力平滑化が範囲外です。")
            if not 0.05 <= settings["scene_cut_threshold"] <= 1.0:
                raise ValueError("シーン変化リセットが範囲外です。")
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

    def _poll_events(self) -> None:
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

    def _update_metrics(self, metrics: dict[str, Any]) -> None:
        self.status_detail_var.set(str(metrics.get("status", "")))
        self.metric_vars["input_fps"].set(f"{metrics.get('input_fps', 0):.1f}")
        self.metric_vars["output_fps"].set(f"{metrics.get('output_fps', 0):.1f}")
        self.metric_vars["inference_fps"].set(f"{metrics.get('inference_fps', 0):.1f}")
        self.metric_vars["inference_ms"].set(f"{metrics.get('inference_ms', 0):.1f} ms")
        self.metric_vars["gpu_utilization"].set(f"{metrics.get('gpu_utilization', 0):.0f} %")
        used = metrics.get("vram_used_mb", 0) / 1024
        total = metrics.get("vram_total_mb", 0) / 1024
        self.metric_vars["vram"].set(f"{used:.1f}/{total:.1f} GB" if total else "-")
        temp = metrics.get("gpu_temperature", 0)
        self.metric_vars["gpu_temperature"].set(f"{temp:.0f} °C" if temp else "-")
        self.metric_vars["input_resolution"].set(str(metrics.get("input_resolution", "-")))
        self.metric_vars["output_resolution"].set(str(metrics.get("output_resolution", "-")))
        self.metric_vars["motion_score"].set(f"{metrics.get('motion_score', 0) * 100:.1f} %")
        self.metric_vars["temporal_feedback"].set(
            f"{metrics.get('temporal_feedback', 0) * 100:.0f} %"
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
        self.start_button.configure(state="disabled" if is_running else "normal")
        self.stop_button.configure(state="normal" if is_running else "disabled")
        self.update_prompt_button.configure(state="normal" if state == "実行中" else "disabled")
        self.reset_temporal_button.configure(state="normal" if state == "実行中" else "disabled")

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
            scene_cut = round(float(self.scene_cut_threshold_var.get()) * 100)
        except tk.TclError:
            return
        self.temporal_feedback_label_var.set(f"入力フレーム保持  {feedback}%")
        self.temporal_smoothing_label_var.set(f"出力平滑化  {smoothing}%")
        self.scene_cut_label_var.set(f"シーン変化リセット  {scene_cut}%")

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
        if self._live_update_after_id is not None:
            self.after_cancel(self._live_update_after_id)
            self._live_update_after_id = None
        self.worker.stop()
        self.destroy()


def run_ui(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    app = DiffusionApp(config_path=config_path)
    app.mainloop()
