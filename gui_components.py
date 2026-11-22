"""Professional GUI components — waveform visualiser, theme, dialogs, and widget builders."""
from __future__ import annotations

import math
from PySide6.QtWidgets import (
    QLabel, QLineEdit, QPushButton, QComboBox, QSlider, QFileDialog,
    QMessageBox, QDialog, QDialogButtonBox, QVBoxLayout, QHBoxLayout,
    QFrame, QSpinBox, QDoubleSpinBox, QSizePolicy, QWidget, QToolBar,
    QStatusBar, QTabWidget, QListWidget, QListWidgetItem, QMenu,
    QProgressBar, QTextEdit, QCheckBox,
)
from PySide6.QtCore import Qt, Signal, QRectF, QPointF, QTimer, QSize
from PySide6.QtGui import (
    QFont, QColor, QPen, QBrush, QPainter, QLinearGradient,
    QPainterPath, QIcon, QAction, QPixmap, QRadialGradient,
)
from config import save_config


# ════════════════════════════════════════════════════════════════════
#  PROFESSIONAL DARK THEME
# ════════════════════════════════════════════════════════════════════

THEME = {
    # Background layers
    'bg_primary':     '#0d0d12',
    'bg_secondary':   '#14141c',
    'bg_card':        '#1a1a26',
    'bg_input':       '#12121c',
    'bg_hover':       '#22223a',
    'bg_selected':    '#2a2a44',

    # Accent colours
    'accent_primary': '#7c5cfc',   # Purple
    'accent_hover':   '#9b7eff',
    'accent_dim':     '#4a3aaa',
    'accent_glow':    '#7c5cfc40',
    'accent_green':   '#34d399',
    'accent_red':     '#f87171',
    'accent_yellow':  '#fbbf24',
    'accent_blue':    '#60a5fa',
    'accent_pink':    '#f472b6',

    # Text
    'text_primary':   '#e4e4f0',
    'text_secondary': '#8888a8',
    'text_muted':     '#555570',
    'text_accent':    '#9b7eff',

    # Borders
    'border_default': '#2a2a3e',
    'border_hover':   '#3a3a58',
    'border_focus':   '#7c5cfc',

    # Misc
    'shadow':         '#00000060',
    'gradient_start': '#7c5cfc',
    'gradient_end':   '#f472b6',
}


def _build_stylesheet() -> str:
    T = THEME
    return f"""
    /* ── Global ─────────────────────────────────────────────────── */
    QMainWindow, QDialog {{
        background-color: {T['bg_primary']};
    }}
    QWidget {{
        color: {T['text_secondary']};
        font-family: 'Inter', 'Segoe UI', 'Arial', sans-serif;
        font-size: 12px;
    }}
    QLabel {{
        color: {T['text_secondary']};
        background: transparent;
    }}

    /* ── Scroll Bars ───────────────────────────────────────────── */
    QScrollBar:vertical {{
        background: {T['bg_primary']};
        width: 8px;
        border-radius: 4px;
    }}
    QScrollBar::handle:vertical {{
        background: {T['border_default']};
        border-radius: 4px;
        min-height: 20px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {T['border_hover']};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}
    QScrollBar:horizontal {{
        background: {T['bg_primary']};
        height: 8px;
        border-radius: 4px;
    }}
    QScrollBar::handle:horizontal {{
        background: {T['border_default']};
        border-radius: 4px;
        min-width: 20px;
    }}

    /* ── Splitter ──────────────────────────────────────────────── */
    QSplitter::handle {{
        background-color: {T['border_default']};
        width: 2px;
    }}
    QSplitter::handle:hover {{
        background-color: {T['accent_primary']};
    }}

    /* ── Text Edit / Console ───────────────────────────────────── */
    QTextEdit {{
        background-color: {T['bg_primary']};
        border: 1px solid {T['border_default']};
        border-radius: 6px;
        padding: 8px;
        color: {T['text_primary']};
        selection-background-color: {T['bg_selected']};
    }}
    QTextEdit:focus {{
        border-color: {T['border_focus']};
    }}

    /* ── Tab Widget ────────────────────────────────────────────── */
    QTabWidget::pane {{
        border: 1px solid {T['border_default']};
        border-radius: 6px;
        background-color: {T['bg_secondary']};
        top: -1px;
    }}
    QTabBar::tab {{
        background-color: {T['bg_card']};
        color: {T['text_muted']};
        border: 1px solid {T['border_default']};
        border-bottom: none;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
        padding: 8px 16px;
        margin-right: 2px;
    }}
    QTabBar::tab:selected {{
        background-color: {T['bg_secondary']};
        color: {T['text_accent']};
        border-color: {T['accent_primary']};
    }}
    QTabBar::tab:hover:!selected {{
        background-color: {T['bg_hover']};
        color: {T['text_primary']};
    }}

    /* ── List Widget ───────────────────────────────────────────── */
    QListWidget {{
        background-color: {T['bg_primary']};
        border: 1px solid {T['border_default']};
        border-radius: 6px;
        padding: 4px;
        outline: none;
    }}
    QListWidget::item {{
        padding: 8px 10px;
        border-radius: 4px;
        color: {T['text_secondary']};
    }}
    QListWidget::item:hover {{
        background-color: {T['bg_hover']};
        color: {T['text_primary']};
    }}
    QListWidget::item:selected {{
        background-color: {T['bg_selected']};
        color: {T['text_accent']};
    }}

    /* ── Check Box ─────────────────────────────────────────────── */
    QCheckBox {{
        spacing: 6px;
        color: {T['text_secondary']};
    }}
    QCheckBox::indicator {{
        width: 16px;
        height: 16px;
        border: 1px solid {T['border_default']};
        border-radius: 4px;
        background-color: {T['bg_input']};
    }}
    QCheckBox::indicator:checked {{
        background-color: {T['accent_primary']};
        border-color: {T['accent_primary']};
    }}

    /* ── Tool Tip ──────────────────────────────────────────────── */
    QToolTip {{
        background-color: {T['bg_card']};
        color: {T['text_primary']};
        border: 1px solid {T['border_hover']};
        border-radius: 6px;
        padding: 6px 10px;
        font-size: 11px;
    }}

    /* ── Menu ──────────────────────────────────────────────────── */
    QMenu {{
        background-color: {T['bg_card']};
        border: 1px solid {T['border_default']};
        border-radius: 8px;
        padding: 4px;
    }}
    QMenu::item {{
        padding: 6px 24px;
        border-radius: 4px;
        color: {T['text_secondary']};
    }}
    QMenu::item:selected {{
        background-color: {T['bg_hover']};
        color: {T['text_primary']};
    }}

    /* ── Status Bar ────────────────────────────────────────────── */
    QStatusBar {{
        background-color: {T['bg_primary']};
        border-top: 1px solid {T['border_default']};
        color: {T['text_muted']};
        font-size: 11px;
    }}
    """


# ── Reusable Widget Stylesheets ──────────────────────────────────────

STYLES = {
    'card': f"""
        QFrame#card {{
            background-color: {THEME['bg_card']};
            border: 1px solid {THEME['border_default']};
            border-radius: 8px;
            padding: 4px;
        }}
    """,

    'slider': f"""
        QSlider::groove:horizontal {{
            border: none;
            height: 4px;
            background: {THEME['border_default']};
            border-radius: 2px;
        }}
        QSlider::handle:horizontal {{
            background: {THEME['accent_primary']};
            border: none;
            width: 14px;
            margin: -5px 0;
            border-radius: 7px;
        }}
        QSlider::handle:horizontal:hover {{
            background: {THEME['accent_hover']};
        }}
        QSlider::sub-page:horizontal {{
            background: {THEME['accent_dim']};
            border-radius: 2px;
        }}
    """,

    'dropdown': f"""
        QComboBox {{
            background-color: {THEME['bg_input']};
            border: 1px solid {THEME['border_default']};
            border-radius: 6px;
            padding: 6px 10px;
            color: {THEME['text_primary']};
            min-height: 22px;
        }}
        QComboBox:hover {{
            border-color: {THEME['border_hover']};
        }}
        QComboBox::drop-down {{
            border: none;
            width: 24px;
        }}
        QComboBox::down-arrow {{
            image: none;
            border: none;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 5px solid {THEME['text_muted']};
            margin-right: 8px;
        }}
        QComboBox QAbstractItemView {{
            background-color: {THEME['bg_card']};
            border: 1px solid {THEME['border_default']};
            border-radius: 6px;
            selection-background-color: {THEME['bg_selected']};
            color: {THEME['text_primary']};
            padding: 4px;
            outline: none;
        }}
        QComboBox QAbstractItemView::item {{
            padding: 8px 12px;
            border-radius: 4px;
        }}
        QComboBox QAbstractItemView::item:hover {{
            background-color: {THEME['bg_hover']};
        }}
    """,

    'spinbox': f"""
        QSpinBox, QDoubleSpinBox {{
            background-color: {THEME['bg_input']};
            border: 1px solid {THEME['border_default']};
            border-radius: 6px;
            padding: 4px 8px;
            color: {THEME['text_primary']};
            min-height: 22px;
        }}
        QSpinBox:hover, QDoubleSpinBox:hover {{
            border-color: {THEME['border_hover']};
        }}
        QSpinBox::up-button, QDoubleSpinBox::up-button,
        QSpinBox::down-button, QDoubleSpinBox::down-button {{
            background-color: {THEME['bg_card']};
            border: none;
            width: 18px;
        }}
        QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {{
            background-color: {THEME['bg_hover']};
        }}
    """,

    'button_primary': f"""
        QPushButton {{
            background-color: {THEME['accent_primary']};
            color: #fff;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: 600;
            font-size: 13px;
        }}
        QPushButton:hover {{
            background-color: {THEME['accent_hover']};
        }}
        QPushButton:pressed {{
            background-color: {THEME['accent_dim']};
        }}
        QPushButton:disabled {{
            background-color: {THEME['bg_card']};
            color: {THEME['text_muted']};
        }}
    """,

    'button_secondary': f"""
        QPushButton {{
            background-color: {THEME['bg_card']};
            color: {THEME['text_secondary']};
            border: 1px solid {THEME['border_default']};
            border-radius: 8px;
            padding: 8px 14px;
            font-size: 12px;
        }}
        QPushButton:hover {{
            background-color: {THEME['bg_hover']};
            border-color: {THEME['border_hover']};
            color: {THEME['text_primary']};
        }}
        QPushButton:disabled {{
            background-color: {THEME['bg_primary']};
            color: {THEME['text_muted']};
        }}
    """,

    'button_danger': f"""
        QPushButton {{
            background-color: #3b1c1c;
            color: {THEME['accent_red']};
            border: 1px solid #5c2a2a;
            border-radius: 8px;
            padding: 8px 14px;
            font-size: 12px;
        }}
        QPushButton:hover {{
            background-color: #4c2222;
            border-color: #7c3a3a;
        }}
        QPushButton:disabled {{
            background-color: {THEME['bg_primary']};
            color: {THEME['text_muted']};
        }}
    """,

    'button_icon': f"""
        QPushButton {{
            background-color: transparent;
            color: {THEME['text_secondary']};
            border: none;
            border-radius: 6px;
            padding: 6px;
            font-size: 16px;
        }}
        QPushButton:hover {{
            background-color: {THEME['bg_hover']};
            color: {THEME['text_primary']};
        }}
    """,

    'line_edit': f"""
        QLineEdit {{
            background-color: {THEME['bg_input']};
            border: 1px solid {THEME['border_default']};
            border-radius: 6px;
            padding: 6px 10px;
            color: {THEME['text_primary']};
        }}
        QLineEdit:focus {{
            border-color: {THEME['accent_primary']};
        }}
    """,

    'progress_bar': f"""
        QProgressBar {{
            border: none;
            border-radius: 4px;
            text-align: center;
            background-color: {THEME['bg_card']};
            color: {THEME['text_primary']};
            height: 20px;
            font-size: 10px;
        }}
        QProgressBar::chunk {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 {THEME['accent_primary']}, stop:1 {THEME['accent_pink']});
            border-radius: 4px;
        }}
    """,
}


def apply_global_styles(app):
    """Apply the professional dark theme stylesheet."""
    app.setStyleSheet(_build_stylesheet())


# ════════════════════════════════════════════════════════════════════
#  WAVEFORM WIDGET
# ════════════════════════════════════════════════════════════════════

class WaveformWidget(QWidget):
    """Real-time waveform visualiser with gradient fill and playhead."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(80)
        self.setMaximumHeight(120)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._peaks = None
        self._play_position = 0.0  # 0.0 – 1.0
        self._is_playing = False
        self._anim_phase = 0.0

        # Animation timer for idle state
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._tick_animation)
        self._anim_timer.start(33)  # ~30 fps

    def set_peaks(self, peaks):
        """Set waveform peak data (0–1 normalised)."""
        self._peaks = peaks
        self.update()

    def set_play_position(self, position: float):
        """Set playback position (0.0 – 1.0)."""
        self._play_position = max(0.0, min(1.0, position))
        self.update()

    def set_playing(self, playing: bool):
        self._is_playing = playing
        self.update()

    def clear(self):
        self._peaks = None
        self._play_position = 0.0
        self.update()

    def _tick_animation(self):
        if self._peaks is None:
            self._anim_phase += 0.04
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        mid_y = h / 2.0

        # Background
        painter.fillRect(0, 0, w, h, QColor(THEME['bg_card']))

        # Grid lines
        pen = QPen(QColor(THEME['border_default']), 1)
        painter.setPen(pen)
        painter.drawLine(0, int(mid_y), w, int(mid_y))
        for frac in (0.25, 0.75):
            painter.setPen(QPen(QColor(THEME['border_default']), 0.5, Qt.DotLine))
            painter.drawLine(0, int(h * frac), w, int(h * frac))

        if self._peaks is not None and len(self._peaks) > 0:
            self._draw_waveform(painter, w, h, mid_y)
        else:
            self._draw_idle_animation(painter, w, h, mid_y)

        # Playhead
        if self._is_playing and self._peaks is not None:
            px = int(w * self._play_position)
            pen = QPen(QColor(THEME['accent_primary']), 2)
            painter.setPen(pen)
            painter.drawLine(px, 0, px, h)

        # Border
        painter.setPen(QPen(QColor(THEME['border_default']), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(0, 0, w - 1, h - 1, 6, 6)
        painter.end()

    def _draw_waveform(self, painter, w, h, mid_y):
        peaks = self._peaks
        n = len(peaks)
        bar_w = max(w / n, 1)
        half_h = (h / 2) - 4

        gradient = QLinearGradient(0, 0, w, 0)
        gradient.setColorAt(0, QColor(THEME['accent_primary']))
        gradient.setColorAt(1, QColor(THEME['accent_pink']))

        for i, peak in enumerate(peaks):
            x = int(i * bar_w)
            bar_h = int(peak * half_h)
            if bar_h < 1:
                bar_h = 1
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(gradient))
            painter.drawRoundedRect(x, int(mid_y - bar_h), max(int(bar_w) - 1, 1), bar_h * 2, 1, 1)

    def _draw_idle_animation(self, painter, w, h, mid_y):
        """Draw gentle sine wave animation when no audio is loaded."""
        half_h = h / 6
        pen = QPen(QColor(THEME['accent_dim']), 2)
        painter.setPen(pen)

        path = QPainterPath()
        for x in range(w):
            y = mid_y + math.sin(x * 0.02 + self._anim_phase) * half_h * \
                (0.5 + 0.5 * math.sin(x * 0.005 + self._anim_phase * 0.3))
            if x == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        painter.drawPath(path)


# ════════════════════════════════════════════════════════════════════
#  LEVEL METER WIDGET
# ════════════════════════════════════════════════════════════════════

class LevelMeter(QWidget):
    """Simple vertical level meter bar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(8)
        self.setMinimumHeight(40)
        self._level = 0.0

    def set_level(self, level: float):
        self._level = max(0.0, min(1.0, level))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        h = self.height()
        w = self.width()

        # Background
        painter.fillRect(0, 0, w, h, QColor(THEME['bg_input']))

        # Level
        level_h = int(h * self._level)
        if level_h > 0:
            gradient = QLinearGradient(0, h, 0, h - level_h)
            gradient.setColorAt(0, QColor(THEME['accent_green']))
            gradient.setColorAt(0.7, QColor(THEME['accent_yellow']))
            gradient.setColorAt(1.0, QColor(THEME['accent_red']))
            painter.fillRect(0, h - level_h, w, level_h, gradient)

        painter.end()


# ════════════════════════════════════════════════════════════════════
#  SETTINGS DIALOG
# ════════════════════════════════════════════════════════════════════

class SettingsDialog(QDialog):
    """Global settings dialog."""

    def __init__(self, current_config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(520)
        self.setMinimumHeight(400)
        self.config = current_config

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        # Title
        title = QLabel("Settings")
        title.setFont(QFont("Inter", 16, QFont.Bold))
        title.setStyleSheet(f"color: {THEME['text_primary']};")
        layout.addWidget(title)

        # Cache Folder
        layout.addWidget(self._create_path_row(
            "Model Cache:",
            self.config.get('system', {}).get('cache_dir', ''),
            'cache',
            "Directory for downloaded model files"
        ))

        # Checkpoints Folder
        layout.addWidget(self._create_path_row(
            "Checkpoints:",
            self.config.get('system', {}).get('ckpt_dir', ''),
            'ckpt',
            "Directory for model weight files"
        ))

        # HeartMuLa Repo
        layout.addWidget(self._create_path_row(
            "HeartMuLa Repo:",
            self.config.get('system', {}).get('heartlib_repo_dir', ''),
            'repo',
            "Path to cloned heartlib repository"
        ))

        # Output Folder
        layout.addWidget(self._create_path_row(
            "Output Folder:",
            self.config.get('settings', {}).get('output_dir', ''),
            'output',
            "Where generated audio files are saved"
        ))

        # Device selector
        device_row = QHBoxLayout()
        device_label = QLabel("Device:")
        device_label.setMinimumWidth(100)
        device_row.addWidget(device_label)
        self.device_combo = QComboBox()
        # Build device list with smart detection
        device_items = ["auto"]
        try:
            from gpu_manager import get_gpu_manager
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
        self.device_combo.setToolTip("auto = smart selection based on VRAM availability")
        current_device = self.config.get('hardware', {}).get('device', 'auto')
        idx = self.device_combo.findText(current_device)
        if idx >= 0:
            self.device_combo.setCurrentIndex(idx)
        else:
            # Default to auto
            idx = self.device_combo.findText("auto")
            if idx >= 0:
                self.device_combo.setCurrentIndex(idx)
        device_row.addWidget(self.device_combo)
        layout.addLayout(device_row)

        # Auto-play checkbox
        self.auto_play_cb = QCheckBox("Auto-play after generation")
        self.auto_play_cb.setChecked(self.config.get('settings', {}).get('auto_play', True))
        layout.addWidget(self.auto_play_cb)

        layout.addStretch()

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(STYLES['button_secondary'])
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        save_btn = QPushButton("Save")
        save_btn.setStyleSheet(STYLES['button_primary'])
        save_btn.clicked.connect(self._save_settings)
        btn_layout.addWidget(save_btn)

        layout.addLayout(btn_layout)

    def _create_path_row(self, label_text, default_value, attr_suffix, tooltip_text):
        frame = QFrame()
        frame.setObjectName("card")
        frame.setStyleSheet(STYLES['card'])
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(8)

        label = QLabel(label_text)
        label.setMinimumWidth(100)
        label.setToolTip(tooltip_text)
        label.setStyleSheet(f"color: {THEME['text_secondary']};")
        layout.addWidget(label)

        edit = QLineEdit(default_value)
        edit.setStyleSheet(STYLES['line_edit'])
        edit.setToolTip(tooltip_text)
        layout.addWidget(edit)

        btn = QPushButton("Browse")
        btn.setStyleSheet(STYLES['button_secondary'])
        btn.setMaximumWidth(70)
        btn.clicked.connect(lambda: self._browse(edit, f"Select {label_text.strip(':')}"))
        layout.addWidget(btn)

        setattr(self, f'{attr_suffix}_edit', edit)
        return frame

    def _browse(self, line_edit, title="Select Directory"):
        folder = QFileDialog.getExistingDirectory(self, title)
        if folder:
            line_edit.setText(folder)

    def _save_settings(self):
        self.config['system']['cache_dir'] = self.cache_edit.text()
        self.config['system']['ckpt_dir'] = self.ckpt_edit.text()
        self.config['system']['heartlib_repo_dir'] = self.repo_edit.text()
        self.config['settings']['output_dir'] = self.output_edit.text()
        self.config['hardware']['device'] = self.device_combo.currentText()
        self.config['settings']['auto_play'] = self.auto_play_cb.isChecked()
        try:
            save_config(self.config)
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save config:\n{e}")


# ════════════════════════════════════════════════════════════════════
#  PRESET DIALOG
# ════════════════════════════════════════════════════════════════════

class PresetDialog(QDialog):
    """Dialog for saving and naming a preset."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Save Preset")
        self.setMinimumWidth(360)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("Save Preset")
        title.setFont(QFont("Inter", 14, QFont.Bold))
        title.setStyleSheet(f"color: {THEME['text_primary']};")
        layout.addWidget(title)

        self.name_edit = QLineEdit()
        self.name_edit.setStyleSheet(STYLES['line_edit'])
        self.name_edit.setPlaceholderText("Enter preset name…")
        layout.addWidget(self.name_edit)

        btn_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(STYLES['button_secondary'])
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        save_btn = QPushButton("Save")
        save_btn.setStyleSheet(STYLES['button_primary'])
        save_btn.clicked.connect(self.accept)
        btn_layout.addWidget(save_btn)
        layout.addLayout(btn_layout)

    def get_name(self) -> str:
        return self.name_edit.text().strip()


# ════════════════════════════════════════════════════════════════════
#  EFFECTS DIALOG
# ════════════════════════════════════════════════════════════════════

class EffectsDialog(QDialog):
    """Post-processing effects configuration dialog."""

    effects_applied = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Audio Effects")
        self.setMinimumWidth(420)
        self.effects = {}

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("Post-Processing Effects")
        title.setFont(QFont("Inter", 14, QFont.Bold))
        title.setStyleSheet(f"color: {THEME['text_primary']};")
        layout.addWidget(title)

        # Normalize
        self.normalize_cb = QCheckBox("Normalize (peak to 0 dBFS)")
        layout.addWidget(self.normalize_cb)

        # Gain
        gain_row = QHBoxLayout()
        gain_row.addWidget(QLabel("Gain (dB):"))
        self.gain_spin = QDoubleSpinBox()
        self.gain_spin.setRange(-20, 20)
        self.gain_spin.setValue(0)
        self.gain_spin.setStyleSheet(STYLES['spinbox'])
        gain_row.addWidget(self.gain_spin)
        gain_row.addStretch()
        layout.addLayout(gain_row)

        # Fade In
        fi_row = QHBoxLayout()
        fi_row.addWidget(QLabel("Fade In (s):"))
        self.fade_in_spin = QDoubleSpinBox()
        self.fade_in_spin.setRange(0, 10)
        self.fade_in_spin.setValue(0)
        self.fade_in_spin.setSingleStep(0.1)
        self.fade_in_spin.setStyleSheet(STYLES['spinbox'])
        fi_row.addWidget(self.fade_in_spin)
        fi_row.addStretch()
        layout.addLayout(fi_row)

        # Fade Out
        fo_row = QHBoxLayout()
        fo_row.addWidget(QLabel("Fade Out (s):"))
        self.fade_out_spin = QDoubleSpinBox()
        self.fade_out_spin.setRange(0, 10)
        self.fade_out_spin.setValue(0)
        self.fade_out_spin.setSingleStep(0.1)
        self.fade_out_spin.setStyleSheet(STYLES['spinbox'])
        fo_row.addWidget(self.fade_out_spin)
        fo_row.addStretch()
        layout.addLayout(fo_row)

        # Reverb
        rev_row = QHBoxLayout()
        rev_row.addWidget(QLabel("Reverb Mix:"))
        self.reverb_slider = QSlider(Qt.Horizontal)
        self.reverb_slider.setRange(0, 100)
        self.reverb_slider.setValue(0)
        self.reverb_slider.setStyleSheet(STYLES['slider'])
        rev_row.addWidget(self.reverb_slider)
        self.reverb_label = QLabel("0%")
        self.reverb_label.setStyleSheet(f"color: {THEME['text_muted']}; min-width: 35px;")
        self.reverb_slider.valueChanged.connect(
            lambda v: self.reverb_label.setText(f"{v}%")
        )
        rev_row.addWidget(self.reverb_label)
        layout.addLayout(rev_row)

        layout.addStretch()

        # Buttons
        btn_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(STYLES['button_secondary'])
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        apply_btn = QPushButton("Apply")
        apply_btn.setStyleSheet(STYLES['button_primary'])
        apply_btn.clicked.connect(self._apply)
        btn_layout.addWidget(apply_btn)
        layout.addLayout(btn_layout)

    def _apply(self):
        self.effects = {
            'normalize': self.normalize_cb.isChecked(),
            'gain_db': self.gain_spin.value(),
            'fade_in': self.fade_in_spin.value(),
            'fade_out': self.fade_out_spin.value(),
            'reverb_mix': self.reverb_slider.value() / 100.0,
        }
        self.effects_applied.emit(self.effects)
        self.accept()

    def get_effects(self) -> dict:
        return self.effects


# ════════════════════════════════════════════════════════════════════
#  SETTINGS WIDGET BUILDERS
# ════════════════════════════════════════════════════════════════════

class SettingWidget:
    """Container for a setting's UI elements and metadata."""

    def __init__(self, setting_id, widget, value_label=None, default=None, step=None):
        self.id = setting_id
        self.widget = widget
        self.value_label = value_label
        self.default = default
        self.step = step

    def get_value(self):
        if isinstance(self.widget, QSlider):
            if self.step and self.step >= 1:
                return self.widget.value()
            else:
                return self.widget.value() / 10.0
        elif isinstance(self.widget, QComboBox):
            return self.widget.currentText()
        elif isinstance(self.widget, QLineEdit):
            return self.widget.text()
        elif isinstance(self.widget, (QSpinBox, QDoubleSpinBox)):
            return self.widget.value()
        return None

    def set_value(self, value):
        if isinstance(self.widget, QSlider):
            if self.step and self.step >= 1:
                self.widget.setValue(int(value))
            else:
                self.widget.setValue(int(float(value) * 10))
            if self.value_label:
                self.value_label.setText(self._format_value(value))
        elif isinstance(self.widget, QComboBox):
            idx = self.widget.findText(str(value))
            if idx >= 0:
                self.widget.setCurrentIndex(idx)
        elif isinstance(self.widget, QLineEdit):
            self.widget.setText(str(value))
        elif isinstance(self.widget, (QSpinBox, QDoubleSpinBox)):
            self.widget.setValue(float(value))

    def _format_value(self, value):
        if self.step and self.step >= 1:
            return str(int(value))
        return f"{float(value):.1f}"

    def reset_to_default(self):
        if self.default is not None:
            self.set_value(self.default)


def build_settings_widgets(schema, layout, settings_widgets: dict,
                           saved_values: dict = None):
    """Build UI widgets for a settings schema, grouped by category."""
    categories = {}
    uncategorized = []

    for item in schema:
        cat = item.get('category')
        if cat:
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(item)
        else:
            uncategorized.append(item)

    for cat_name, items in categories.items():
        cat_header = QLabel(cat_name.upper())
        cat_header.setStyleSheet(
            f"color: {THEME['text_muted']}; font-size: 10px; font-weight: 600;"
            f"letter-spacing: 1px; padding: 8px 0 2px 0;"
        )
        layout.addWidget(cat_header)

        # Separator line
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {THEME['border_default']}; max-height: 1px;")
        layout.addWidget(sep)

        for item in items:
            _build_single_setting(item, layout, settings_widgets, saved_values)

    for item in uncategorized:
        _build_single_setting(item, layout, settings_widgets, saved_values)


def _build_single_setting(item, layout, settings_widgets: dict, saved_values: dict):
    """Build a single setting widget."""
    setting_id = item['id']
    label = item.get('label', setting_id)
    description = item.get('description', '')
    default = item.get('default')
    saved_value = saved_values.get(setting_id, default) if saved_values else default

    card = QFrame()
    card.setObjectName("card")
    card.setStyleSheet(STYLES['card'])
    card_layout = QHBoxLayout(card)
    card_layout.setContentsMargins(10, 6, 10, 6)
    card_layout.setSpacing(8)

    label_widget = QLabel(label)
    label_widget.setMinimumWidth(85)
    label_widget.setStyleSheet(f"color: {THEME['text_secondary']};")
    if description:
        label_widget.setToolTip(description)
    card_layout.addWidget(label_widget)

    if item['type'] == 'slider':
        widget, value_label = _build_slider(item, saved_value)
        card_layout.addWidget(widget)
        card_layout.addWidget(value_label)
        settings_widgets[setting_id] = SettingWidget(
            setting_id, widget, value_label, default, item.get('step')
        )
    elif item['type'] == 'dropdown':
        widget = _build_dropdown(item, saved_value)
        card_layout.addWidget(widget)
        settings_widgets[setting_id] = SettingWidget(
            setting_id, widget, default=default
        )
    elif item['type'] == 'text':
        widget = _build_text_input(item, saved_value)
        card_layout.addWidget(widget)
        settings_widgets[setting_id] = SettingWidget(
            setting_id, widget, default=default
        )
    elif item['type'] == 'spinbox':
        widget = _build_spinbox(item, saved_value)
        card_layout.addWidget(widget)
        settings_widgets[setting_id] = SettingWidget(
            setting_id, widget, default=default
        )

    layout.addWidget(card)


def _build_slider(item, saved_value):
    min_val = item.get('min', 0)
    max_val = item.get('max', 100)
    step = item.get('step', 0.1)
    default = item.get('default', min_val)
    value = saved_value if saved_value is not None else default

    is_int_step = step >= 1
    if is_int_step:
        scale = 1
        slider_min, slider_max, slider_val = int(min_val), int(max_val), int(value)
    else:
        scale = 10
        slider_min = int(min_val * scale)
        slider_max = int(max_val * scale)
        slider_val = int(value * scale)

    slider = QSlider(Qt.Horizontal)
    slider.setMinimum(slider_min)
    slider.setMaximum(slider_max)
    slider.setValue(slider_val)
    slider.setStyleSheet(STYLES['slider'])

    def format_val(v):
        return f"{v}" if is_int_step else f"{v / scale:.1f}"

    value_label = QLabel(format_val(slider_val))
    value_label.setStyleSheet(
        f"color: {THEME['accent_primary']}; font-family: 'Consolas', monospace;"
        f"font-size: 11px; min-width: 40px; font-weight: 600;"
    )
    value_label.setAlignment(Qt.AlignCenter)
    slider.valueChanged.connect(lambda v: value_label.setText(format_val(v)))

    return slider, value_label


def _build_dropdown(item, saved_value):
    combo = QComboBox()
    combo.addItems(item.get('options', []))
    if saved_value is not None:
        idx = combo.findText(str(saved_value))
        if idx >= 0:
            combo.setCurrentIndex(idx)
    combo.setStyleSheet(STYLES['dropdown'])
    if item.get('description'):
        combo.setToolTip(item['description'])
    return combo


def _build_text_input(item, saved_value):
    edit = QLineEdit(str(saved_value) if saved_value is not None else item.get('default', ''))
    edit.setStyleSheet(STYLES['line_edit'])
    edit.setPlaceholderText(item.get('placeholder', ''))
    if item.get('description'):
        edit.setToolTip(item['description'])
    return edit


def _build_spinbox(item, saved_value):
    min_val = item.get('min', 0)
    max_val = item.get('max', 1000)
    step = item.get('step', 1)
    default = item.get('default', min_val)
    value = saved_value if saved_value is not None else default

    if step >= 1:
        spinbox = QSpinBox()
        spinbox.setMinimum(int(min_val))
        spinbox.setMaximum(int(max_val))
        spinbox.setSingleStep(int(step))
        spinbox.setValue(int(value))
    else:
        spinbox = QDoubleSpinBox()
        spinbox.setMinimum(float(min_val))
        spinbox.setMaximum(float(max_val))
        spinbox.setSingleStep(float(step))
        spinbox.setDecimals(1)
        spinbox.setValue(float(value))

    spinbox.setStyleSheet(STYLES['spinbox'])
    return spinbox


def create_reset_button(settings_widgets: dict, parent_layout):
    btn = QPushButton("Reset Defaults")
    btn.setStyleSheet(STYLES['button_secondary'])

    def reset_all():
        for widget in settings_widgets.values():
            widget.reset_to_default()

    btn.clicked.connect(reset_all)
    parent_layout.addWidget(btn, alignment=Qt.AlignRight)
    return btn


def clear_layout(layout):
    """Safely clear a layout and delete all child widgets."""
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget:
            widget.deleteLater()
        sub_layout = item.layout()
        if sub_layout:
            clear_layout(sub_layout)


# ════════════════════════════════════════════════════════════════════
#  GPU MONITOR WIDGET
# ════════════════════════════════════════════════════════════════════

class GPUVRamBar(QWidget):
    """A horizontal VRAM usage bar with gradient fill and label."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(20)
        self.setMinimumWidth(120)
        self._usage_pct = 0.0
        self._used_mb = 0
        self._total_mb = 0
        self._color_ok = THEME['accent_green']
        self._color_warn = THEME['accent_yellow']
        self._color_full = THEME['accent_red']

    def set_values(self, used_mb: int, total_mb: int):
        self._used_mb = used_mb
        self._total_mb = total_mb
        self._usage_pct = (used_mb / total_mb * 100) if total_mb > 0 else 0
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        bar_h = 8
        y = (h - bar_h) // 2

        # Background track
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(THEME['bg_input']))
        painter.drawRoundedRect(0, y, w, bar_h, 4, 4)

        # Fill bar with gradient based on usage
        fill_w = int(w * self._usage_pct / 100)
        if fill_w > 0:
            if self._usage_pct < 60:
                color = self._color_ok
            elif self._usage_pct < 85:
                color = self._color_warn
            else:
                color = self._color_full

            gradient = QLinearGradient(0, 0, fill_w, 0)
            gradient.setColorAt(0, QColor(color))
            gradient.setColorAt(1, QColor(color).darker(120))
            painter.setBrush(gradient)
            painter.drawRoundedRect(0, y, fill_w, bar_h, 4, 4)

        # Text label
        painter.setPen(QColor(THEME['text_primary']))
        font = QFont("Consolas", 8)
        painter.setFont(font)
        used_gb = self._used_mb / 1024
        total_gb = self._total_mb / 1024
        text = f"{used_gb:.1f}/{total_gb:.1f} GB"
        painter.drawText(self.rect(), Qt.AlignCenter, text)

        painter.end()


class GPUMonitorWidget(QFrame):
    """Compact GPU status card with VRAM bar, temperature, and device info.

    Polls the GPUManager at regular intervals and updates the display.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("gpuMonitor")
        self.setStyleSheet(f"""
            QFrame#gpuMonitor {{
                background-color: {THEME['bg_card']};
                border: 1px solid {THEME['border_default']};
                border-radius: 8px;
                padding: 2px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        # Header
        header_row = QHBoxLayout()
        header_row.setSpacing(4)

        self.gpu_icon = QLabel("GPU")
        self.gpu_icon.setStyleSheet(
            f"color: {THEME['accent_primary']}; font-size: 10px; font-weight: 700;"
            f"letter-spacing: 1px;"
        )
        header_row.addWidget(self.gpu_icon)
        header_row.addStretch()

        self.device_label = QLabel("—")
        self.device_label.setStyleSheet(f"color: {THEME['text_muted']}; font-size: 10px;")
        header_row.addWidget(self.device_label)
        layout.addLayout(header_row)

        # GPU name
        self.gpu_name_label = QLabel("Detecting…")
        self.gpu_name_label.setStyleSheet(f"color: {THEME['text_primary']}; font-size: 11px; font-weight: 600;")
        self.gpu_name_label.setWordWrap(True)
        layout.addWidget(self.gpu_name_label)

        # VRAM bar
        self.vram_bar = GPUVRamBar()
        layout.addWidget(self.vram_bar)

        # Stats row
        stats_row = QHBoxLayout()
        stats_row.setSpacing(8)

        self.temp_label = QLabel("")
        self.temp_label.setStyleSheet(f"color: {THEME['text_muted']}; font-size: 9px;")
        stats_row.addWidget(self.temp_label)

        self.util_label = QLabel("")
        self.util_label.setStyleSheet(f"color: {THEME['text_muted']}; font-size: 9px;")
        stats_row.addWidget(self.util_label)

        self.driver_label = QLabel("")
        self.driver_label.setStyleSheet(f"color: {THEME['text_muted']}; font-size: 9px;")
        stats_row.addWidget(self.driver_label)

        stats_row.addStretch()
        layout.addLayout(stats_row)

        # Recommendation row
        self.rec_label = QLabel("")
        self.rec_label.setWordWrap(True)
        self.rec_label.setStyleSheet(f"color: {THEME['accent_blue']}; font-size: 9px; padding: 2px 0;")
        layout.addWidget(self.rec_label)

        # Refresh timer
        self._timer = QTimer(self)
        self._timer.setInterval(3000)  # Refresh every 3s
        self._timer.timeout.connect(self.refresh)
        self._timer.start()
        self._current_model_vram_gb = 0.0

        # Initial refresh
        QTimer.singleShot(200, self.refresh)

    def set_model_vram_requirement(self, vram_gb: float):
        """Set the current model's VRAM requirement for the recommendation."""
        self._current_model_vram_gb = vram_gb
        self._update_recommendation()

    def refresh(self):
        """Refresh GPU stats from the GPU manager."""
        try:
            from gpu_manager import get_gpu_manager
            gm = get_gpu_manager()
            gm.refresh_vram()

            if gm.gpus:
                gpu = gm.gpus[0]  # Primary GPU
                self.gpu_name_label.setText(gpu.short_name)
                self.vram_bar.set_values(gpu.used_vram_mb, gpu.total_vram_mb)
                self.device_label.setText(f"Device: {gpu.device_type.value}:{gpu.index}")

                # Temperature
                if gpu.temperature_c > 0:
                    temp_color = THEME['accent_green'] if gpu.temperature_c < 70 else (
                        THEME['accent_yellow'] if gpu.temperature_c < 85 else THEME['accent_red']
                    )
                    self.temp_label.setText(f"🌡 {gpu.temperature_c}°C")
                    self.temp_label.setStyleSheet(f"color: {temp_color}; font-size: 9px;")

                # Utilisation
                if gpu.utilisation_pct > 0:
                    self.util_label.setText(f"⚡ {gpu.utilisation_pct}%")
                    self.util_label.setStyleSheet(f"color: {THEME['text_muted']}; font-size: 9px;")

                # Driver
                if gpu.driver_version:
                    self.driver_label.setText(f"Driver {gpu.driver_version}")
                    self.driver_label.setStyleSheet(f"color: {THEME['text_muted']}; font-size: 9px;")

                self._update_recommendation()
            else:
                self.gpu_name_label.setText("No GPU Detected")
                self.vram_bar.set_values(0, 0)
                self.device_label.setText("CPU Only")
                self.temp_label.setText("")
                self.util_label.setText("")
                self.driver_label.setText("")
                self.rec_label.setText("All generation will run on CPU — consider installing CUDA for faster inference.")
                self.rec_label.setStyleSheet(f"color: {THEME['accent_yellow']}; font-size: 9px; padding: 2px 0;")

        except Exception as e:
            self.gpu_name_label.setText(f"Detection error: {str(e)[:40]}")

    def _update_recommendation(self):
        """Update the device recommendation label."""
        if self._current_model_vram_gb <= 0:
            self.rec_label.setText("")
            return
        try:
            from gpu_manager import get_gpu_manager
            gm = get_gpu_manager()
            rec = gm.recommend_device(vram_needed_gb=self._current_model_vram_gb)
            if rec.vram_sufficient:
                self.rec_label.setText(f"✓ {rec.device.upper()} — {rec.reason}")
                self.rec_label.setStyleSheet(f"color: {THEME['accent_green']}; font-size: 9px; padding: 2px 0;")
            elif rec.device != "cpu":
                self.rec_label.setText(f"⚠ {rec.reason}")
                self.rec_label.setStyleSheet(f"color: {THEME['accent_yellow']}; font-size: 9px; padding: 2px 0;")
            else:
                self.rec_label.setText(f"⚠ {rec.reason}")
                self.rec_label.setStyleSheet(f"color: {THEME['accent_red']}; font-size: 9px; padding: 2px 0;")
        except Exception:
            pass
