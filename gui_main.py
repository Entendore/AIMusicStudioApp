"""Professional AI Music Studio — main application window with 3-panel layout and Model Hub."""
from __future__ import annotations

import os
import sys
import logging
from typing import Optional

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QSpinBox, QPushButton, QTextEdit,
    QSplitter, QMessageBox, QProgressBar, QComboBox, QScrollArea,
    QFrame, QTabWidget, QListWidget, QListWidgetItem, QMenu,
    QFileDialog, QStatusBar, QToolBar, QSizePolicy, QCheckBox,
    QSlider, QGraphicsDropShadowEffect,
)
from PySide6.QtCore import Qt, QThread, Slot, QUrl, QTimer, QSize, QPoint
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtGui import QFont, QAction, QIcon, QKeySequence, QColor, QPainter

from core import PluginManager, save_audio, export_audio, compute_waveform_peaks
from config import (
    save_config, save_preset, load_preset, delete_preset,
    list_presets, add_to_history, get_history, clear_history,
)
from gui_components import (
    SettingsDialog, PresetDialog, EffectsDialog,
    WaveformWidget, LevelMeter, GPUMonitorWidget,
    build_settings_widgets, clear_layout,
    apply_global_styles, STYLES, THEME, SettingWidget, create_reset_button,
)
from gui_workers import Worker, ExportWorker, EffectsWorker, LogEmitter, QtLogHandler
from model_manager import (
    ModelInfo, ModelStatus, InstallMethod, MODEL_REGISTRY,
    refresh_all_statuses, get_model_by_id, get_installed_models,
)
from model_hub_dialog import ModelHubDialog
from gpu_manager import get_gpu_manager, GPUManager

logger = logging.getLogger(__name__)


# ── Backend Metadata (kept for backwards compat) ────────────────────

BACKEND_INFO = {m.id: {
    'name': m.name, 'description': m.description,
    'details': m.details, 'icon': m.icon,
    'category': m.category, 'requirements': m.requirements_text,
    'color': m.color,
} for m in MODEL_REGISTRY}


# ════════════════════════════════════════════════════════════════════
#  SIDEBAR NAVIGATION
# ════════════════════════════════════════════════════════════════════

class SidebarButton(QPushButton):
    """A flat icon+text navigation button for the sidebar."""

    def __init__(self, icon_text: str, label: str, parent=None):
        super().__init__(parent)
        self._icon_text = icon_text
        self._label = label
        self._active = False
        self.setCheckable(True)
        self.setMinimumHeight(44)
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._apply_style()

    def set_active(self, active: bool):
        self._active = active
        self.setChecked(active)
        self._apply_style()

    def _apply_style(self):
        T = THEME
        if self._active:
            bg = T['bg_selected']
            fg = T['accent_primary']
            border_left = f"3px solid {T['accent_primary']}"
        else:
            bg = "transparent"
            fg = T['text_secondary']
            border_left = "3px solid transparent"

        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg};
                color: {fg};
                border: none;
                border-left: {border_left};
                border-radius: 0;
                padding: 10px 14px;
                text-align: left;
                font-size: 12px;
                font-weight: {"600" if self._active else "400"};
            }}
            QPushButton:hover {{
                background-color: {T['bg_hover']};
                color: {T['text_primary']};
            }}
        """)
        self.setText(f"  {self._icon_text}  {self._label}")


# ════════════════════════════════════════════════════════════════════
#  MODEL STATUS BANNER
# ════════════════════════════════════════════════════════════════════

class ModelStatusBanner(QFrame):
    """Compact banner showing installed/available model counts."""

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config
        self.setObjectName("statusBanner")
        self.setStyleSheet(f"""
            QFrame#statusBanner {{
                background-color: {THEME['bg_card']};
                border: 1px solid {THEME['border_default']};
                border-radius: 8px;
                padding: 2px;
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(12)

        self.installed_label = QLabel()
        self.installed_label.setStyleSheet(f"color: {THEME['accent_green']}; font-weight: 600; font-size: 11px;")
        layout.addWidget(self.installed_label)

        self.available_label = QLabel()
        self.available_label.setStyleSheet(f"color: {THEME['accent_primary']}; font-size: 11px;")
        layout.addWidget(self.available_label)

        layout.addStretch()

        hub_btn = QPushButton("Open Model Hub")
        hub_btn.setStyleSheet(STYLES['button_primary'])
        hub_btn.setCursor(Qt.PointingHandCursor)
        hub_btn.clicked.connect(self._open_hub)
        layout.addWidget(hub_btn)

        self._hub_dialog = None
        self.refresh()

    def refresh(self):
        models = refresh_all_statuses(self.config)
        installed = sum(1 for m in models if m.status == ModelStatus.INSTALLED)
        available = len(models) - installed
        self.installed_label.setText(f"  {installed} Models Installed")
        self.available_label.setText(f"{available} Available")

    def _open_hub(self):
        if self._hub_dialog and self._hub_dialog.isVisible():
            self._hub_dialog.raise_()
            return
        self._hub_dialog = ModelHubDialog(self.config, self)
        self._hub_dialog.finished.connect(self.refresh)
        self._hub_dialog.show()


# ════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ════════════════════════════════════════════════════════════════════

class MusicStudioApp(QMainWindow):
    """Professional AI Music Studio main window — 3-panel layout."""

    # Page indices
    PAGE_GENERATE = 0
    PAGE_MODELS   = 1
    PAGE_HISTORY  = 2
    PAGE_SETTINGS = 3

    def __init__(self, manager: PluginManager, config: dict):
        super().__init__()
        self.manager = manager
        self.config = config
        self.thread = None
        self.worker = None
        self.settings_widgets: dict[str, SettingWidget] = {}
        self.last_audio_data: Optional[dict] = None
        self.last_saved_path: Optional[str] = None
        self._current_filename: Optional[str] = None

        self.setWindowTitle("AI Music Studio Pro")
        self.resize(1280, 820)
        self.setMinimumSize(960, 640)

        # ── Media Player (must be before UI build) ────────────────
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.player.playbackStateChanged.connect(self._update_playback_state)
        self.player.positionChanged.connect(self._update_play_position)
        self.player.durationChanged.connect(self._update_duration)
        self._playback_duration = 0
        self._position_timer = QTimer(self)
        self._position_timer.setInterval(50)

        # ── Central Widget ────────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── LEFT: Sidebar ─────────────────────────────────────────
        self.sidebar = QFrame()
        self.sidebar.setFixedWidth(200)
        self.sidebar.setStyleSheet(f"""
            QFrame {{
                background-color: {THEME['bg_secondary']};
                border-right: 1px solid {THEME['border_default']};
            }}
        """)
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        # App logo area
        logo_frame = QFrame()
        logo_frame.setStyleSheet(f"background-color: {THEME['bg_primary']}; border: none;")
        logo_layout = QVBoxLayout(logo_frame)
        logo_layout.setContentsMargins(16, 16, 16, 10)
        logo_layout.setSpacing(2)

        logo_text = QLabel("AI Music Studio")
        logo_text.setFont(QFont("Inter", 13, QFont.Bold))
        logo_text.setStyleSheet(f"color: {THEME['accent_primary']};")
        logo_layout.addWidget(logo_text)

        logo_sub = QLabel("Pro Edition")
        logo_sub.setStyleSheet(f"color: {THEME['text_muted']}; font-size: 10px; letter-spacing: 2px;")
        logo_layout.addWidget(logo_sub)

        sidebar_layout.addWidget(logo_frame)

        # Nav buttons
        self.nav_buttons: list[SidebarButton] = []
        nav_items = [
            ("🎵", "Generate", self.PAGE_GENERATE),
            ("📦", "Model Hub", self.PAGE_MODELS),
            ("🕐", "History", self.PAGE_HISTORY),
            ("⚙", "Settings", self.PAGE_SETTINGS),
        ]
        for icon, label, page_idx in nav_items:
            btn = SidebarButton(icon, label)
            btn.clicked.connect(lambda checked, idx=page_idx: self._switch_page(idx))
            sidebar_layout.addWidget(btn)
            self.nav_buttons.append(btn)

        self.nav_buttons[0].set_active(True)

        # Model status banner at bottom of sidebar
        sidebar_layout.addStretch()
        self.model_status_banner = ModelStatusBanner(config)
        sidebar_layout.addWidget(self.model_status_banner)

        # GPU monitor in sidebar
        self.gpu_monitor = GPUMonitorWidget()
        sidebar_layout.addWidget(self.gpu_monitor)

        root_layout.addWidget(self.sidebar)

        # ── CENTRE: Content Stack ─────────────────────────────────
        self.content_stack = QFrame()
        self.content_stack.setStyleSheet(f"background-color: {THEME['bg_primary']};")
        self.stack_layout = QVBoxLayout(self.content_stack)
        self.stack_layout.setContentsMargins(0, 0, 0, 0)
        self.stack_layout.setSpacing(0)

        # Create all pages
        self._pages: list[QWidget] = []
        self._build_generate_page()
        self._build_models_page()
        self._build_history_page()
        self._build_settings_page()

        # Show default page
        self._switch_page(self.PAGE_GENERATE)

        root_layout.addWidget(self.content_stack, stretch=1)

        # ── RIGHT: Console Panel ──────────────────────────────────
        self.console_panel = QFrame()
        self.console_panel.setFixedWidth(340)
        self.console_panel.setStyleSheet(f"""
            QFrame {{
                background-color: {THEME['bg_primary']};
                border-left: 1px solid {THEME['border_default']};
            }}
        """)
        console_layout = QVBoxLayout(self.console_panel)
        console_layout.setContentsMargins(8, 8, 8, 8)
        console_layout.setSpacing(4)

        # Console header
        ch = QHBoxLayout()
        ch_label = QLabel("CONSOLE")
        ch_label.setStyleSheet(
            f"color: {THEME['text_muted']}; font-size: 10px; font-weight: 600;"
            f"letter-spacing: 1.5px; padding: 4px;"
        )
        ch.addWidget(ch_label)
        ch.addStretch()
        clear_btn = QPushButton("Clear")
        clear_btn.setStyleSheet(STYLES['button_secondary'])
        clear_btn.clicked.connect(lambda: self.log_text.clear())
        ch.addWidget(clear_btn)
        console_layout.addLayout(ch)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {THEME['bg_primary']};
                color: {THEME['text_muted']};
                font-family: 'JetBrains Mono', 'Consolas', 'Courier New', monospace;
                font-size: 10px;
                border: 1px solid {THEME['border_default']};
                border-radius: 6px;
                padding: 6px;
            }}
        """)
        console_layout.addWidget(self.log_text)

        root_layout.addWidget(self.console_panel)

        # ── Toolbar ───────────────────────────────────────────────
        self._build_toolbar()

        # ── Status Bar ────────────────────────────────────────────
        self._build_status_bar()

        # ── Logging ───────────────────────────────────────────────
        self._setup_logging()

        # ── Load default backend ──────────────────────────────────
        QApplication.processEvents()
        self.load_backend()
        self._refresh_history()
        self._refresh_presets()

    # ════════════════════════════════════════════════════════════════
    #  PAGE SWITCHING
    # ════════════════════════════════════════════════════════════════

    def _switch_page(self, index: int):
        # Hide all pages
        for page in self._pages:
            page.hide()
        # Show selected
        if 0 <= index < len(self._pages):
            self._pages[index].show()
        # Update nav buttons
        for i, btn in enumerate(self.nav_buttons):
            btn.set_active(i == index)

    # ════════════════════════════════════════════════════════════════
    #  PAGE: GENERATE
    # ════════════════════════════════════════════════════════════════

    def _build_generate_page(self):
        page = QWidget()
        page_layout = QHBoxLayout(page)
        page_layout.setContentsMargins(16, 16, 16, 16)
        page_layout.setSpacing(12)

        # ── Left column: Controls ─────────────────────────────────
        left_col = QVBoxLayout()
        left_col.setSpacing(10)

        # Header with gradient
        header = QFrame()
        header.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {THEME['accent_primary']}, stop:1 {THEME['accent_pink']});
                border-radius: 10px;
            }}
        """)
        header_inner = QVBoxLayout(header)
        header_inner.setContentsMargins(16, 12, 16, 12)
        header_inner.setSpacing(2)

        title = QLabel("Generate Music")
        title.setFont(QFont("Inter", 16, QFont.Bold))
        title.setStyleSheet("color: #fff; background: transparent;")
        header_inner.addWidget(title)
        subtitle = QLabel("Select an AI engine, configure parameters, and create")
        subtitle.setStyleSheet("color: #ccc; font-size: 11px; background: transparent;")
        header_inner.addWidget(subtitle)
        left_col.addWidget(header)

        # Backend selector
        self._build_backend_selector_inline(left_col)

        # Input section
        self._build_input_section_inline(left_col)

        # Parameters section
        self._build_settings_section_inline(left_col)

        left_col.addStretch()
        page_layout.addLayout(left_col, stretch=3)

        # ── Right column: Playback ────────────────────────────────
        right_col = QVBoxLayout()
        right_col.setSpacing(10)

        # Controls: Generate/Stop + Progress
        self._build_controls_section_inline(right_col)

        # Playback
        self._build_playback_section_inline(right_col)

        # Presets
        self._build_presets_section_inline(right_col)

        right_col.addStretch()
        page_layout.addLayout(right_col, stretch=2)

        self._pages.append(page)
        self.stack_layout.addWidget(page)

    # ── Inline Builders (Generate Page) ───────────────────────────

    def _build_backend_selector_inline(self, parent_layout):
        section = QFrame()
        section.setObjectName("card")
        section.setStyleSheet(STYLES['card'])
        layout = QVBoxLayout(section)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        hdr = QLabel("ENGINE")
        hdr.setStyleSheet(
            f"color: {THEME['text_muted']}; font-size: 10px; font-weight: 600;"
            f"letter-spacing: 1.5px;"
        )
        layout.addWidget(hdr)

        self.backend_combo = QComboBox()
        self._available_backends = self.manager.discover_available_backends()
        for backend_id in self._available_backends:
            info = BACKEND_INFO.get(backend_id, {})
            icon = info.get('icon', '📦')
            name = info.get('name', backend_id.title())
            self.backend_combo.addItem(f"{icon}  {name}", backend_id)
        self.backend_combo.setStyleSheet(STYLES['dropdown'])

        current_plugin = self.config.get('plugins', {}).get('active_plugin', 'mathsynth')
        for i in range(self.backend_combo.count()):
            if self.backend_combo.itemData(i) == current_plugin:
                self.backend_combo.setCurrentIndex(i)
                break
        self.backend_combo.currentIndexChanged.connect(self._on_backend_selection_changed)
        layout.addWidget(self.backend_combo)

        self.backend_status_label = QLabel("")
        self.backend_status_label.setStyleSheet(f"font-size: 10px; color: {THEME['text_muted']};")
        layout.addWidget(self.backend_status_label)

        self.backend_desc_label = QLabel("")
        self.backend_desc_label.setWordWrap(True)
        self.backend_desc_label.setStyleSheet(f"font-size: 10px; color: {THEME['text_muted']};")
        layout.addWidget(self.backend_desc_label)

        self._update_backend_info(current_plugin)
        parent_layout.addWidget(section)

    def _build_input_section_inline(self, parent_layout):
        self.prompt_group = QFrame()
        self.prompt_group.setObjectName("card")
        self.prompt_group.setStyleSheet(STYLES['card'])
        pg_layout = QVBoxLayout(self.prompt_group)
        pg_layout.setContentsMargins(12, 10, 12, 10)
        pg_layout.setSpacing(6)

        hdr = QLabel("PROMPT")
        hdr.setStyleSheet(
            f"color: {THEME['text_muted']}; font-size: 10px; font-weight: 600;"
            f"letter-spacing: 1.5px;"
        )
        pg_layout.addWidget(hdr)

        self.prompt_desc_label = QLabel()
        self.prompt_desc_label.setWordWrap(True)
        self.prompt_desc_label.setStyleSheet(f"color: {THEME['text_muted']}; font-size: 10px;")
        pg_layout.addWidget(self.prompt_desc_label)

        self.prompt_input = QTextEdit()
        self.prompt_input.setMinimumHeight(80)
        self.prompt_input.setMaximumHeight(160)
        self.prompt_input.setStyleSheet(f"""
            QTextEdit {{
                background-color: {THEME['bg_input']};
                border: 1px solid {THEME['border_default']};
                border-radius: 8px;
                padding: 8px;
                color: {THEME['text_primary']};
                font-size: 12px;
            }}
            QTextEdit:focus {{ border-color: {THEME['accent_primary']}; }}
        """)
        pg_layout.addWidget(self.prompt_input)

        row = QHBoxLayout()
        row.addWidget(QLabel("Duration:"))
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(1, 600)
        self.duration_spin.setValue(30)
        self.duration_spin.setSuffix("s")
        self.duration_spin.setStyleSheet(STYLES['spinbox'])
        self.duration_spin.setMaximumWidth(90)
        row.addWidget(self.duration_spin)
        row.addStretch()
        pg_layout.addLayout(row)

        parent_layout.addWidget(self.prompt_group)

    def _build_settings_section_inline(self, parent_layout):
        section = QFrame()
        section.setObjectName("card")
        section.setStyleSheet(STYLES['card'])
        layout = QVBoxLayout(section)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)

        hdr = QLabel("PARAMETERS")
        hdr.setStyleSheet(
            f"color: {THEME['text_muted']}; font-size: 10px; font-weight: 600;"
            f"letter-spacing: 1.5px;"
        )
        layout.addWidget(hdr)

        self.settings_scroll = QScrollArea()
        self.settings_scroll.setWidgetResizable(True)
        self.settings_scroll.setMaximumHeight(220)
        self.settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.settings_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self.settings_container = QWidget()
        self.settings_layout = QVBoxLayout(self.settings_container)
        self.settings_layout.setSpacing(4)
        self.settings_layout.setContentsMargins(0, 0, 0, 0)
        self.settings_scroll.setWidget(self.settings_container)
        layout.addWidget(self.settings_scroll)

        parent_layout.addWidget(section)

    def _build_controls_section_inline(self, parent_layout):
        section = QFrame()
        section.setObjectName("card")
        section.setStyleSheet(STYLES['card'])
        layout = QVBoxLayout(section)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        # Generate button — gradient and large
        self.btn_generate = QPushButton("Generate")
        self.btn_generate.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {THEME['accent_primary']}, stop:1 {THEME['accent_pink']});
                color: #fff; border: none; border-radius: 10px;
                padding: 14px 20px; font-weight: 700; font-size: 14px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {THEME['accent_hover']}, stop:1 {THEME['accent_pink']});
            }}
            QPushButton:disabled {{
                background-color: {THEME['bg_card']}; color: {THEME['text_muted']};
            }}
        """)
        self.btn_generate.setCursor(Qt.PointingHandCursor)
        self.btn_generate.clicked.connect(self.start_generation)
        layout.addWidget(self.btn_generate)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet(STYLES['button_danger'])
        self.btn_stop.setCursor(Qt.PointingHandCursor)
        self.btn_stop.clicked.connect(self.stop_generation)
        layout.addWidget(self.btn_stop)

        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(STYLES['progress_bar'])
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Action row
        row = QHBoxLayout()
        row.setSpacing(4)
        self.btn_export = QPushButton("Export")
        self.btn_export.setStyleSheet(STYLES['button_secondary'])
        self.btn_export.setCursor(Qt.PointingHandCursor)
        self.btn_export.clicked.connect(self._export_audio)
        self.btn_export.setEnabled(False)
        row.addWidget(self.btn_export)

        self.btn_effects = QPushButton("Effects")
        self.btn_effects.setStyleSheet(STYLES['button_secondary'])
        self.btn_effects.setCursor(Qt.PointingHandCursor)
        self.btn_effects.clicked.connect(self._open_effects)
        self.btn_effects.setEnabled(False)
        row.addWidget(self.btn_effects)
        layout.addLayout(row)

        parent_layout.addWidget(section)

    def _build_playback_section_inline(self, parent_layout):
        section = QFrame()
        section.setObjectName("card")
        section.setStyleSheet(STYLES['card'])
        layout = QVBoxLayout(section)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        hdr_row = QHBoxLayout()
        hdr = QLabel("PLAYBACK")
        hdr.setStyleSheet(
            f"color: {THEME['text_muted']}; font-size: 10px; font-weight: 600;"
            f"letter-spacing: 1.5px;"
        )
        hdr_row.addWidget(hdr)
        hdr_row.addStretch()
        self.lbl_now_playing = QLabel("No audio")
        self.lbl_now_playing.setStyleSheet(f"color: {THEME['text_muted']}; font-size: 10px;")
        hdr_row.addWidget(self.lbl_now_playing)
        layout.addLayout(hdr_row)

        # Waveform
        self.waveform = WaveformWidget()
        layout.addWidget(self.waveform)

        # Time
        time_row = QHBoxLayout()
        self.lbl_current_time = QLabel("0:00")
        self.lbl_current_time.setStyleSheet(
            f"color: {THEME['text_muted']}; font-family: 'Consolas'; font-size: 10px;"
        )
        time_row.addWidget(self.lbl_current_time)
        time_row.addStretch()
        self.lbl_total_time = QLabel("0:00")
        self.lbl_total_time.setStyleSheet(
            f"color: {THEME['text_muted']}; font-family: 'Consolas'; font-size: 10px;"
        )
        time_row.addWidget(self.lbl_total_time)
        layout.addLayout(time_row)

        # Volume
        vol_row = QHBoxLayout()
        vol_row.addWidget(QLabel("Vol:"))
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(80)
        self.volume_slider.setStyleSheet(STYLES['slider'])
        self.volume_slider.setMaximumWidth(100)
        self.volume_slider.valueChanged.connect(self._update_volume)
        vol_row.addWidget(self.volume_slider)
        self.volume_label = QLabel("80%")
        self.volume_label.setStyleSheet(f"color: {THEME['text_muted']}; font-size: 10px; min-width: 30px;")
        vol_row.addWidget(self.volume_label)
        self.level_meter = LevelMeter()
        vol_row.addWidget(self.level_meter)
        layout.addLayout(vol_row)

        # Transport
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(4)
        self.btn_play = QPushButton("Play")
        self.btn_play.setEnabled(False)
        self.btn_play.setMinimumHeight(30)
        self.btn_play.setStyleSheet(STYLES['button_primary'])
        self.btn_play.setCursor(Qt.PointingHandCursor)
        self.btn_play.clicked.connect(self.play_audio)
        ctrl_row.addWidget(self.btn_play)
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setEnabled(False)
        self.btn_pause.setMinimumHeight(30)
        self.btn_pause.setStyleSheet(STYLES['button_secondary'])
        self.btn_pause.clicked.connect(self.pause_audio)
        ctrl_row.addWidget(self.btn_pause)
        self.btn_stop_play = QPushButton("Stop")
        self.btn_stop_play.setEnabled(False)
        self.btn_stop_play.setMinimumHeight(30)
        self.btn_stop_play.setStyleSheet(STYLES['button_secondary'])
        self.btn_stop_play.clicked.connect(self.stop_audio)
        ctrl_row.addWidget(self.btn_stop_play)
        layout.addLayout(ctrl_row)

        self._update_volume(80)
        parent_layout.addWidget(section)

    def _build_presets_section_inline(self, parent_layout):
        section = QFrame()
        section.setObjectName("card")
        section.setStyleSheet(STYLES['card'])
        layout = QVBoxLayout(section)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)

        hdr_row = QHBoxLayout()
        hdr = QLabel("PRESETS")
        hdr.setStyleSheet(
            f"color: {THEME['text_muted']}; font-size: 10px; font-weight: 600;"
            f"letter-spacing: 1.5px;"
        )
        hdr_row.addWidget(hdr)
        hdr_row.addStretch()

        save_btn = QPushButton("Save")
        save_btn.setStyleSheet(STYLES['button_secondary'])
        save_btn.setCursor(Qt.PointingHandCursor)
        save_btn.clicked.connect(self._save_current_preset)
        hdr_row.addWidget(save_btn)
        layout.addLayout(hdr_row)

        self.preset_list = QListWidget()
        self.preset_list.setMaximumHeight(120)
        self.preset_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.preset_list.customContextMenuRequested.connect(self._preset_context_menu)
        self.preset_list.itemDoubleClicked.connect(self._load_preset_item)
        layout.addWidget(self.preset_list)

        parent_layout.addWidget(section)

    # ════════════════════════════════════════════════════════════════
    #  PAGE: MODEL HUB
    # ════════════════════════════════════════════════════════════════

    def _build_models_page(self):
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(0)

        # Header
        header = QFrame()
        header.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {THEME['accent_primary']}22, stop:1 {THEME['bg_primary']});
                border-bottom: 1px solid {THEME['border_default']};
            }}
        """)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(24, 20, 24, 16)

        title_col = QVBoxLayout()
        title_col.setSpacing(2)
        title = QLabel("Model Hub")
        title.setFont(QFont("Inter", 20, QFont.Bold))
        title.setStyleSheet(f"color: {THEME['text_primary']};")
        title_col.addWidget(title)
        sub = QLabel("Download and install AI music models to unlock generation capabilities")
        sub.setStyleSheet(f"color: {THEME['text_secondary']}; font-size: 12px;")
        sub.setWordWrap(True)
        title_col.addWidget(sub)
        header_layout.addLayout(title_col)

        header_layout.addStretch()

        self.models_installed_count = QLabel()
        self.models_installed_count.setFont(QFont("Inter", 24, QFont.Bold))
        self.models_installed_count.setStyleSheet(f"color: {THEME['accent_primary']};")
        header_layout.addWidget(self.models_installed_count)

        page_layout.addWidget(header)

        # Model cards in scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        cards_container = QWidget()
        self.models_cards_layout = QVBoxLayout(cards_container)
        self.models_cards_layout.setContentsMargins(24, 16, 24, 16)
        self.models_cards_layout.setSpacing(12)

        scroll.setWidget(cards_container)
        page_layout.addWidget(scroll)

        # Refresh button at bottom
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(24, 8, 24, 12)
        btn_row.addStretch()

        refresh_btn = QPushButton("Refresh Model Status")
        refresh_btn.setStyleSheet(STYLES['button_secondary'])
        refresh_btn.clicked.connect(self._refresh_model_cards)
        btn_row.addWidget(refresh_btn)

        open_hub_btn = QPushButton("Open Full Model Hub")
        open_hub_btn.setStyleSheet(STYLES['button_primary'])
        open_hub_btn.setCursor(Qt.PointingHandCursor)
        open_hub_btn.clicked.connect(self._open_model_hub)
        btn_row.addWidget(open_hub_btn)

        page_layout.addLayout(btn_row)

        self._model_cards: dict[str, 'ModelCardWidget'] = {}
        self._model_hub_dialog = None

        self._pages.append(page)
        self.stack_layout.addWidget(page)

    def _refresh_model_cards(self):
        """Rebuild model cards on the Model Hub page."""
        # Clear existing
        for card in self._model_cards.values():
            card.deleteLater()
        self._model_cards.clear()

        from model_hub_dialog import ModelCardWidget
        models = refresh_all_statuses(self.config)
        installed = sum(1 for m in models if m.status == ModelStatus.INSTALLED)

        self.models_installed_count.setText(f"{installed}/{len(models)}")

        # Group by category
        from model_manager import get_models_by_category
        categories = get_models_by_category(self.config)

        for cat_name, cat_models in sorted(categories.items()):
            # Category header
            cat_header = QLabel(f"  {cat_name}")
            cat_header.setFont(QFont("Inter", 12, QFont.Bold))
            cat_header.setStyleSheet(
                f"color: {THEME['text_secondary']}; padding: 8px 0 4px 0;"
            )
            self.models_cards_layout.addWidget(cat_header)

            sep = QFrame()
            sep.setFrameShape(QFrame.HLine)
            sep.setStyleSheet(f"color: {THEME['border_default']}; max-height: 1px;")
            self.models_cards_layout.addWidget(sep)

            for model in cat_models:
                card = ModelCardWidget(model)
                card.install_requested.connect(self._open_model_hub)
                self._model_cards[model.id] = card
                self.models_cards_layout.addWidget(card)

        self.models_cards_layout.addStretch()

        # Also update sidebar banner
        self.model_status_banner.refresh()

    def _open_model_hub(self, model_id: str = None):
        if self._model_hub_dialog and self._model_hub_dialog.isVisible():
            self._model_hub_dialog.raise_()
            return
        self._model_hub_dialog = ModelHubDialog(self.config, self)
        self._model_hub_dialog.finished.connect(self._on_hub_closed)
        self._model_hub_dialog.show()

    def _on_hub_closed(self):
        self._refresh_model_cards()

    # ════════════════════════════════════════════════════════════════
    #  PAGE: HISTORY
    # ════════════════════════════════════════════════════════════════

    def _build_history_page(self):
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(24, 20, 24, 16)
        page_layout.setSpacing(12)

        title = QLabel("Generation History")
        title.setFont(QFont("Inter", 20, QFont.Bold))
        title.setStyleSheet(f"color: {THEME['text_primary']};")
        page_layout.addWidget(title)

        self.history_list = QListWidget()
        self.history_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.history_list.customContextMenuRequested.connect(self._history_context_menu)
        self.history_list.itemDoubleClicked.connect(self._play_history_item)
        page_layout.addWidget(self.history_list)

        btn_row = QHBoxLayout()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setStyleSheet(STYLES['button_secondary'])
        refresh_btn.clicked.connect(self._refresh_history)
        btn_row.addWidget(refresh_btn)
        clear_btn = QPushButton("Clear All")
        clear_btn.setStyleSheet(STYLES['button_danger'])
        clear_btn.clicked.connect(self._clear_all_history)
        btn_row.addWidget(clear_btn)
        btn_row.addStretch()
        page_layout.addLayout(btn_row)

        self._pages.append(page)
        self.stack_layout.addWidget(page)

    # ════════════════════════════════════════════════════════════════
    #  PAGE: SETTINGS
    # ════════════════════════════════════════════════════════════════

    def _build_settings_page(self):
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(24, 20, 24, 16)
        page_layout.setSpacing(12)

        title = QLabel("Settings")
        title.setFont(QFont("Inter", 20, QFont.Bold))
        title.setStyleSheet(f"color: {THEME['text_primary']};")
        page_layout.addWidget(title)

        # Open full settings dialog
        open_btn = QPushButton("Open Settings Dialog")
        open_btn.setStyleSheet(STYLES['button_primary'])
        open_btn.setCursor(Qt.PointingHandCursor)
        open_btn.clicked.connect(self.open_settings)
        page_layout.addWidget(open_btn)

        # Quick settings
        quick_frame = QFrame()
        quick_frame.setObjectName("card")
        quick_frame.setStyleSheet(STYLES['card'])
        qf_layout = QVBoxLayout(quick_frame)
        qf_layout.setContentsMargins(16, 12, 16, 12)
        qf_layout.setSpacing(8)

        qh = QLabel("QUICK SETTINGS")
        qh.setStyleSheet(
            f"color: {THEME['text_muted']}; font-size: 10px; font-weight: 600;"
            f"letter-spacing: 1.5px;"
        )
        qf_layout.addWidget(qh)

        # Auto-play
        self.auto_play_cb = QCheckBox("Auto-play after generation")
        self.auto_play_cb.setChecked(self.config.get('settings', {}).get('auto_play', True))
        self.auto_play_cb.stateChanged.connect(
            lambda v: self.config.setdefault('settings', {}).__setitem__('auto_play', bool(v))
        )
        qf_layout.addWidget(self.auto_play_cb)

        # Device — smart GPU detection
        dev_row = QHBoxLayout()
        dev_row.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        device_items = ["auto"]
        try:
            gm = get_gpu_manager()
            if gm.has_cuda:
                for gpu in gm.gpus:
                    if gpu.device_type.value == "cuda":
                        device_items.append(f"cuda:{gpu.index}")
                device_items.append("cuda")
            if gm.has_mps:
                device_items.append("mps")
        except Exception:
            device_items.extend(["cuda", "mps"])
        device_items.append("cpu")
        self.device_combo.addItems(device_items)
        self.device_combo.setStyleSheet(STYLES['dropdown'])
        self.device_combo.setToolTip("auto = smart GPU selection based on VRAM")
        cur = self.config.get('hardware', {}).get('device', 'auto')
        idx = self.device_combo.findText(cur)
        if idx >= 0:
            self.device_combo.setCurrentIndex(idx)
        else:
            auto_idx = self.device_combo.findText("auto")
            if auto_idx >= 0:
                self.device_combo.setCurrentIndex(auto_idx)
        dev_row.addWidget(self.device_combo)

        # GPU info label
        self.quick_gpu_label = QLabel("")
        self.quick_gpu_label.setStyleSheet(f"color: {THEME['text_muted']}; font-size: 10px;")
        dev_row.addWidget(self.quick_gpu_label)
        # Populate GPU info
        try:
            gm = get_gpu_manager()
            if gm.gpus:
                g = gm.gpus[0]
                self.quick_gpu_label.setText(f"({g.short_name}, {g.free_vram_gb:.1f} GB free)")
        except Exception:
            pass

        dev_row.addStretch()
        qf_layout.addLayout(dev_row)

        page_layout.addWidget(quick_frame)
        page_layout.addStretch()

        self._pages.append(page)
        self.stack_layout.addWidget(page)

    # ════════════════════════════════════════════════════════════════
    #  TOOLBAR
    # ════════════════════════════════════════════════════════════════

    def _build_toolbar(self):
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(18, 18))
        toolbar.setStyleSheet(f"""
            QToolBar {{
                background-color: {THEME['bg_secondary']};
                border-bottom: 1px solid {THEME['border_default']};
                padding: 2px 8px; spacing: 2px;
            }}
            QToolButton {{
                background: transparent; color: {THEME['text_secondary']};
                border: none; border-radius: 6px; padding: 5px 8px; font-size: 11px;
            }}
            QToolButton:hover {{
                background-color: {THEME['bg_hover']}; color: {THEME['text_primary']};
            }}
        """)

        gen_action = QAction("Generate", self)
        gen_action.setShortcut(QKeySequence("Ctrl+G"))
        gen_action.triggered.connect(self.start_generation)
        toolbar.addAction(gen_action)

        stop_action = QAction("Stop", self)
        stop_action.triggered.connect(self.stop_generation)
        toolbar.addAction(stop_action)

        toolbar.addSeparator()

        hub_action = QAction("Model Hub", self)
        hub_action.setShortcut(QKeySequence("Ctrl+M"))
        hub_action.triggered.connect(lambda: self._switch_page(self.PAGE_MODELS))
        toolbar.addAction(hub_action)

        toolbar.addSeparator()

        export_action = QAction("Export", self)
        export_action.setShortcut(QKeySequence("Ctrl+E"))
        export_action.triggered.connect(self._export_audio)
        toolbar.addAction(export_action)

        effects_action = QAction("Effects", self)
        effects_action.triggered.connect(self._open_effects)
        toolbar.addAction(effects_action)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(spacer)

        brand = QLabel("AI Music Studio Pro")
        brand.setFont(QFont("Inter", 10, QFont.Bold))
        brand.setStyleSheet(f"color: {THEME['accent_primary']}; padding-right: 8px;")
        toolbar.addWidget(brand)

        self.addToolBar(toolbar)

    # ════════════════════════════════════════════════════════════════
    #  STATUS BAR
    # ════════════════════════════════════════════════════════════════

    def _build_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_backend = QLabel()
        self.status_bar.addWidget(self.status_backend)
        self.status_device = QLabel()
        self.status_bar.addPermanentWidget(self.status_device)

    # ════════════════════════════════════════════════════════════════
    #  LOGGING
    # ════════════════════════════════════════════════════════════════

    def _setup_logging(self):
        self.log_emitter = LogEmitter()
        self.log_emitter.log_message.connect(self.append_log)
        handler = QtLogHandler(self.log_emitter)
        handler.setFormatter(
            logging.Formatter('%(asctime)s — %(message)s', datefmt='%H:%M:%S')
        )
        logging.getLogger().addHandler(handler)

    # ════════════════════════════════════════════════════════════════
    #  BACKEND
    # ════════════════════════════════════════════════════════════════

    def _on_backend_selection_changed(self, index):
        if index < 0:
            return
        backend_name = self.backend_combo.itemData(index)
        if not backend_name:
            return
        if self.thread and self.thread.isRunning():
            QMessageBox.warning(self, "Busy", "Stop generation before switching backends.")
            current = self.config.get('plugins', {}).get('active_plugin', 'mathsynth')
            self.backend_combo.blockSignals(True)
            for i in range(self.backend_combo.count()):
                if self.backend_combo.itemData(i) == current:
                    self.backend_combo.setCurrentIndex(i)
            self.backend_combo.blockSignals(False)
            return
        if self.manager.active_backend:
            self.manager.active_backend.unload()
        self.config['plugins']['active_plugin'] = backend_name
        save_config(self.config)
        self._update_backend_info(backend_name)
        self.load_backend()

    def _update_backend_info(self, backend_name):
        info = BACKEND_INFO.get(backend_name, {})
        name = info.get('name', backend_name.title())
        desc = info.get('description', '')
        details = info.get('details', '')
        color = info.get('color', THEME['text_primary'])
        self.backend_status_label.setText(f"Loading {name}…")
        self.backend_status_label.setStyleSheet(f"font-size: 11px; color: {color}; font-weight: 600;")
        self.backend_desc_label.setText(f"{desc} — {details}")

    def open_settings(self):
        SettingsDialog(self.config, self).exec()

    def populate_settings(self):
        clear_layout(self.settings_layout)
        self.settings_widgets = {}
        if not self.manager.active_backend:
            lbl = QLabel("No backend loaded")
            lbl.setStyleSheet(f"color: {THEME['text_muted']}; font-size: 11px; padding: 16px;")
            lbl.setAlignment(Qt.AlignCenter)
            self.settings_layout.addWidget(lbl)
            self._configure_prompt_section(None)
            return
        prompt_schema = self.manager.active_backend.__class__.get_prompt_schema()
        self._configure_prompt_section(prompt_schema)
        schema = self.manager.active_backend.__class__.get_settings_schema()
        backend_name = self.config.get('plugins', {}).get('active_plugin', '')
        saved = self.config.get('backend_settings', {}).get(backend_name, {})
        build_settings_widgets(schema, self.settings_layout, self.settings_widgets, saved)
        create_reset_button(self.settings_widgets, self.settings_layout)

    def save_backend_settings(self):
        if not self.settings_widgets:
            return
        backend_name = self.config.get('plugins', {}).get('active_plugin', '')
        if not backend_name:
            return
        self.config.setdefault('backend_settings', {}).setdefault(backend_name, {})
        for key, widget in self.settings_widgets.items():
            value = widget.get_value()
            if value is not None:
                self.config['backend_settings'][backend_name][key] = value
        save_config(self.config)

    def load_backend(self):
        plugin_name = self.config['plugins']['active_plugin']
        info = BACKEND_INFO.get(plugin_name, {})
        name = info.get('name', plugin_name.title())

        # Smart GPU device selection
        model_info = get_model_by_id(plugin_name)
        vram_needed = model_info.vram_requirement_gb if model_info else 0.0
        self.gpu_monitor.set_model_vram_requirement(vram_needed)

        gm = get_gpu_manager()
        smart_device = gm.get_device_for_config(self.config)
        self.config['hardware']['device'] = smart_device

        # Check if model can fit in VRAM and warn
        if vram_needed > 0 and gm.gpus:
            rec = gm.recommend_device(vram_needed_gb=vram_needed, model_name=name)
            if not rec.vram_sufficient and rec.device == "cpu":
                reply = QMessageBox.warning(
                    self, "Low VRAM Warning",
                    f"{name} requires ~{vram_needed:.0f} GB VRAM but no GPU has enough free memory.\n\n"
                    f"{rec.reason}\n\n"
                    f"Continue with CPU? (Generation will be much slower)",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if reply == QMessageBox.No:
                    self.btn_generate.setEnabled(False)
                    self.backend_status_label.setText(f"  {name} — VRAM insufficient")
                    self.backend_status_label.setStyleSheet(
                        f"font-size: 11px; color: {THEME['accent_yellow']}; font-weight: 600;"
                    )
                    return
            elif not rec.vram_sufficient:
                # GPU exists but tight — just warn in log
                logging.warning(f"VRAM tight for {name}: {rec.reason}")

        # Release GPU memory from previous model before loading new one
        gm.release_memory()

        try:
            self.manager.load_plugin(plugin_name, self.config)
            self.populate_settings()
            self.btn_generate.setEnabled(True)
            device_display = smart_device.upper() if smart_device != "cpu" else "CPU"
            self.backend_status_label.setText(f"  {name} loaded on {device_display}")
            self.backend_status_label.setStyleSheet(
                f"font-size: 11px; color: {THEME['accent_green']}; font-weight: 600;"
            )
            self.status_backend.setText(f"  Engine: {name}")
            self.status_device.setText(f"Device: {smart_device}  ")
        except Exception as e:
            logging.error(f"Failed to load: {e}")
            self.btn_generate.setEnabled(False)
            self.backend_status_label.setText(f"  {name} failed")
            self.backend_status_label.setStyleSheet(
                f"font-size: 11px; color: {THEME['accent_red']}; font-weight: 600;"
            )

    def reload_backend(self):
        if self.manager.active_backend:
            self.manager.active_backend.unload()
        # Release GPU memory before reload
        get_gpu_manager().release_memory()
        self.load_backend()

    def _configure_prompt_section(self, schema):
        if not schema or not schema.get('required', True):
            self.prompt_input.hide()
            self.prompt_desc_label.hide()
            return
        self.prompt_input.show()
        desc = schema.get('description', '')
        if desc:
            self.prompt_desc_label.setText(desc)
            self.prompt_desc_label.show()
        else:
            self.prompt_desc_label.hide()
        self.prompt_input.setPlaceholderText(schema.get('placeholder', ''))
        self.prompt_input.clear()

    # ════════════════════════════════════════════════════════════════
    #  GENERATION
    # ════════════════════════════════════════════════════════════════

    def start_generation(self):
        if self.thread and self.thread.isRunning():
            return
        self.stop_audio()
        self.save_backend_settings()
        params = {k: w.get_value() for k, w in self.settings_widgets.items() if w.get_value() is not None}
        prompt = self.prompt_input.toPlainText()
        prompt_schema = None
        if self.manager.active_backend:
            prompt_schema = self.manager.active_backend.__class__.get_prompt_schema()
        if (prompt_schema or {}).get('required', True) and not prompt.strip():
            QMessageBox.warning(self, "Empty Prompt", "A prompt is required for this backend.")
            return
        self.thread = QThread()
        self.worker = Worker(self.manager.active_backend, prompt, self.duration_spin.value(), params, self.config)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run_generation)
        self.worker.finished.connect(self.on_finished)
        self.worker.success.connect(self.on_success)
        self.worker.error.connect(self.on_error)
        self.worker.progress.connect(self.update_progress)
        self.thread.start()
        self.btn_generate.setEnabled(False)
        self.btn_generate.setText("Generating…")
        self.btn_stop.setEnabled(True)
        self.progress_bar.setValue(0)

    def stop_generation(self):
        if self.worker:
            self.worker.stop()

    # ════════════════════════════════════════════════════════════════
    #  PLAYBACK
    # ════════════════════════════════════════════════════════════════

    def play_audio(self):
        if self.last_saved_path:
            self.player.setSource(QUrl.fromLocalFile(self.last_saved_path))
            self.player.play()
            self._position_timer.start()

    def pause_audio(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        elif self.player.playbackState() == QMediaPlayer.PlaybackState.PausedState:
            self.player.play()

    def stop_audio(self):
        self.player.stop()
        self._position_timer.stop()
        self.waveform.set_playing(False)
        self.waveform.set_play_position(0.0)

    def _update_volume(self, v):
        self.audio_output.setVolume(v / 100.0)
        self.volume_label.setText(f"{v}%")

    def _update_playback_state(self, state):
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        paused = state == QMediaPlayer.PlaybackState.PausedState
        has = self.last_saved_path is not None
        self.btn_play.setEnabled(not playing and has)
        self.btn_pause.setEnabled(playing or paused)
        self.btn_stop_play.setEnabled(playing or paused)
        self.waveform.set_playing(playing)
        if playing:
            fn = os.path.basename(self.last_saved_path) if self.last_saved_path else ""
            self.lbl_now_playing.setText(f"  {fn}")
            self.lbl_now_playing.setStyleSheet(f"color: {THEME['accent_green']}; font-size: 10px;")
        elif paused:
            self.lbl_now_playing.setText("Paused")
            self.lbl_now_playing.setStyleSheet(f"color: {THEME['accent_yellow']}; font-size: 10px;")
        elif has:
            self.lbl_now_playing.setText("Stopped")
            self.lbl_now_playing.setStyleSheet(f"color: {THEME['text_muted']}; font-size: 10px;")

    def _update_play_position(self, pos):
        if self._playback_duration > 0:
            self.waveform.set_play_position(pos / self._playback_duration)
            self.lbl_current_time.setText(self._format_time(pos))
            import random
            if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                self.level_meter.set_level(random.uniform(0.3, 0.9))
            else:
                self.level_meter.set_level(0.0)

    def _update_duration(self, d):
        self._playback_duration = d
        self.lbl_total_time.setText(self._format_time(d))

    @staticmethod
    def _format_time(ms):
        s = ms // 1000
        m, s = divmod(s, 60)
        return f"{m}:{s:02d}"

    # ════════════════════════════════════════════════════════════════
    #  EXPORT & EFFECTS
    # ════════════════════════════════════════════════════════════════

    def _export_audio(self):
        if not self.last_audio_data:
            QMessageBox.information(self, "No Audio", "Generate audio first.")
            return
        fmt_filter = "WAV (*.wav);;FLAC (*.flac);;OGG (*.ogg)"
        path, sel = QFileDialog.getSaveFileName(self, "Export Audio", "", fmt_filter)
        if not path:
            return
        fmt = "wav"
        for k, v in {"WAV": "wav", "FLAC": "flac", "OGG": "ogg"}.items():
            if k in sel:
                fmt = v
        try:
            p = export_audio(self.last_audio_data, os.path.splitext(os.path.basename(path))[0],
                             os.path.dirname(path), fmt)
            self.append_log(f"Exported → {p}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _open_effects(self):
        if not self.last_audio_data:
            QMessageBox.information(self, "No Audio", "Generate audio first.")
            return
        dialog = EffectsDialog(self)
        dialog.effects_applied.connect(self._apply_effects)
        dialog.exec()

    def _apply_effects(self, effects):
        if not self.last_audio_data:
            return
        from core import apply_effects
        try:
            processed = apply_effects(self.last_audio_data['audio'], self.last_audio_data['sample_rate'], effects)
            self.last_audio_data = {'audio': processed, 'sample_rate': self.last_audio_data['sample_rate']}
            output_dir = self.config.get('settings', {}).get('output_dir', 'output')
            if self._current_filename:
                path = save_audio(self.last_audio_data, self._current_filename + "_fx", output_dir)
                self.last_saved_path = path
                self._update_waveform()
                self.append_log(f"Effects applied → {path}")
        except Exception as e:
            QMessageBox.critical(self, "Effects Error", str(e))

    def _update_waveform(self):
        if self.last_audio_data:
            self.waveform.set_peaks(compute_waveform_peaks(self.last_audio_data['audio']))
        else:
            self.waveform.clear()

    # ════════════════════════════════════════════════════════════════
    #  PRESETS
    # ════════════════════════════════════════════════════════════════

    def _save_current_preset(self):
        dialog = PresetDialog(self)
        if dialog.exec() == QDialog.Accepted:
            name = dialog.get_name()
            if not name:
                return
            save_preset(self.config, name, self.config.get('plugins', {}).get('active_plugin', ''),
                        self.prompt_input.toPlainText(), self.duration_spin.value(),
                        {k: w.get_value() for k, w in self.settings_widgets.items()})
            self._refresh_presets()

    def _load_preset_item(self, item):
        name = item.text()
        preset = load_preset(self.config, name)
        if not preset:
            return
        backend = preset.get('backend', '')
        if backend and backend != self.config.get('plugins', {}).get('active_plugin', ''):
            for i in range(self.backend_combo.count()):
                if self.backend_combo.itemData(i) == backend:
                    self.backend_combo.setCurrentIndex(i)
        self.prompt_input.setPlainText(preset.get('prompt', ''))
        self.duration_spin.setValue(preset.get('duration', 30))
        for k, v in preset.get('params', {}).items():
            if k in self.settings_widgets:
                self.settings_widgets[k].set_value(v)

    def _preset_context_menu(self, pos):
        item = self.preset_list.itemAt(pos)
        if not item:
            return
        from PySide6.QtWidgets import QMenu
        menu = QMenu(self)
        menu.addAction("Load", lambda: self._load_preset_item(item))
        menu.addAction("Delete", lambda: (delete_preset(self.config, item.text()), self._refresh_presets()))
        menu.exec(self.preset_list.mapToGlobal(pos))

    def _refresh_presets(self):
        self.preset_list.clear()
        for name in list_presets(self.config):
            self.preset_list.addItem(name)

    # ════════════════════════════════════════════════════════════════
    #  HISTORY
    # ════════════════════════════════════════════════════════════════

    def _refresh_history(self):
        self.history_list.clear()
        for entry in get_history(self.config):
            fp = entry.get('filepath', '')
            backend = entry.get('backend', '')
            ts = entry.get('timestamp', '')[:19]
            info = BACKEND_INFO.get(backend, {})
            icon = info.get('icon', '🎵')
            display = f"{icon}  {os.path.basename(fp)}  —  {ts}"
            item = QListWidgetItem(display)
            item.setToolTip(f"Path: {fp}\nBackend: {info.get('name', backend)}")
            item.setData(Qt.UserRole, fp)
            if not os.path.isfile(fp):
                item.setForeground(QColor(THEME['text_muted']))
            self.history_list.addItem(item)

    def _play_history_item(self, item):
        fp = item.data(Qt.UserRole)
        if fp and os.path.isfile(fp):
            self.last_saved_path = fp
            self.player.setSource(QUrl.fromLocalFile(fp))
            self.player.play()
            self._position_timer.start()
            self.btn_play.setEnabled(True)

    def _history_context_menu(self, pos):
        item = self.history_list.itemAt(pos)
        if not item:
            return
        from PySide6.QtWidgets import QMenu
        menu = QMenu(self)
        menu.addAction("Play", lambda: self._play_history_item(item))
        menu.addAction("Remove", lambda: self.history_list.takeItem(self.history_list.row(item)))
        menu.exec(self.history_list.mapToGlobal(pos))

    def _clear_all_history(self):
        if QMessageBox.question(self, "Clear?", "Clear all history?", QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            clear_history(self.config)
            self._refresh_history()

    # ════════════════════════════════════════════════════════════════
    #  SLOTS
    # ════════════════════════════════════════════════════════════════

    @Slot(str)
    def append_log(self, message):
        if "ERROR" in message or "error" in message.lower() or "failed" in message.lower():
            color = THEME['accent_red']
        elif "WARNING" in message or "warning" in message.lower():
            color = THEME['accent_yellow']
        elif "SUCCESS" in message or "✓" in message or "loaded" in message.lower():
            color = THEME['accent_green']
        elif "Starting" in message or "generating" in message.lower():
            color = THEME['accent_primary']
        else:
            color = THEME['text_muted']
        self.log_text.append(f'<span style="color:{color};">{message}</span>')

    @Slot(int)
    def update_progress(self, v):
        self.progress_bar.setValue(v)

    @Slot(dict, str)
    def on_success(self, result, filename):
        output_dir = self.config.get('settings', {}).get('output_dir', 'output')
        try:
            path = save_audio(result, filename, output_dir)
            self.last_saved_path = path
            self.last_audio_data = result
            self._current_filename = filename
            self.btn_play.setEnabled(True)
            self.btn_export.setEnabled(True)
            self.btn_effects.setEnabled(True)
            self._update_waveform()
            self.append_log(f"✓ Saved: {path}")
            backend = self.config.get('plugins', {}).get('active_plugin', '')
            add_to_history(self.config, path, backend, self.prompt_input.toPlainText(), self.duration_spin.value())
            self._refresh_history()
            if self.config.get('settings', {}).get('auto_play', True):
                self.play_audio()
        except Exception as e:
            logging.error(f"Save error: {e}")

    @Slot(str)
    def on_error(self, msg):
        QMessageBox.critical(self, "Error", f"Generation failed:\n\n{msg}")

    @Slot()
    def on_finished(self):
        if self.thread:
            self.thread.quit()
            self.thread.wait()
            self.thread.deleteLater()
            self.worker.deleteLater()
            self.thread = None
            self.worker = None
        self.btn_generate.setEnabled(True)
        self.btn_generate.setText("Generate")
        self.btn_stop.setEnabled(False)

    def closeEvent(self, event):
        self.save_backend_settings()
        self.stop_audio()
        if self.thread and self.thread.isRunning():
            self.worker.stop()
            self.thread.quit()
            self.thread.wait()
        # Release GPU memory on exit
        try:
            get_gpu_manager().release_all_memory()
        except Exception:
            pass
        event.accept()


# ════════════════════════════════════════════════════════════════════
#  LAUNCH
# ════════════════════════════════════════════════════════════════════

def launch_gui(manager: PluginManager, config: dict):
    app = QApplication(sys.argv)
    app.setApplicationName("AI Music Studio Pro")
    app.setApplicationVersion("2.0.0")
    app.setStyle("Fusion")
    apply_global_styles(app)
    window = MusicStudioApp(manager, config)
    window.show()
    # Build model cards on first show
    window._refresh_model_cards()
    sys.exit(app.exec())
