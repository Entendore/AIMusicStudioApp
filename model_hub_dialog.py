"""Professional Model Hub Dialog — browse, install, and manage AI models."""
from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QProgressBar, QTextEdit, QTabWidget,
    QSizePolicy, QGridLayout, QSpacerItem, QWidget,
)
from PySide6.QtCore import Qt, Signal, Slot, QThread, QTimer, QSize
from PySide6.QtGui import QFont, QPainter, QColor, QPen, QLinearGradient

from gui_components import THEME, STYLES, apply_global_styles
from model_manager import (
    ModelInfo, ModelStatus, InstallMethod,
    MODEL_REGISTRY, refresh_all_statuses, get_install_description,
    get_models_by_category,
)
from model_download_worker import ModelDownloadWorker

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
#  CIRCULAR PROGRESS INDICATOR
# ════════════════════════════════════════════════════════════════════

class CircularProgressWidget(QWidget):
    """A circular progress indicator for model downloads."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(48, 48)
        self._progress = 0
        self._color = THEME['accent_primary']

    def set_progress(self, value: int, color: str = None):
        self._progress = max(0, min(100, value))
        if color:
            self._color = color
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        size = self.width()
        margin = 4
        pen_width = 4

        # Background circle
        painter.setPen(QPen(QColor(THEME['border_default']), pen_width))
        painter.drawArc(margin, margin, size - 2*margin, size - 2*margin, 0, 360 * 16)

        # Progress arc
        if self._progress > 0:
            painter.setPen(QPen(QColor(self._color), pen_width, Qt.SolidLine, Qt.RoundCap))
            span = int(360 * 16 * self._progress / 100)
            painter.drawArc(margin, margin, size - 2*margin, size - 2*margin, 90 * 16, -span)

        # Center text
        painter.setPen(QColor(THEME['text_primary']))
        font = QFont("Inter", 9, QFont.Bold)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignCenter, f"{self._progress}%")

        painter.end()


# ════════════════════════════════════════════════════════════════════
#  MODEL CARD WIDGET
# ════════════════════════════════════════════════════════════════════

class ModelCardWidget(QFrame):
    """A rich card representing a single AI model/engine."""

    install_requested = Signal(str)   # model_id
    uninstall_requested = Signal(str) # model_id

    def __init__(self, model: ModelInfo, parent=None):
        super().__init__(parent)
        self.model = model
        self.setObjectName("modelCard")
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFixedHeight(160)
        self._build_ui()

    def _build_ui(self):
        m = self.model
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(14)

        # ── Left: Icon + Category badge ───────────────────────────
        left_col = QVBoxLayout()
        left_col.setSpacing(4)
        left_col.setAlignment(Qt.AlignTop)

        icon_label = QLabel(m.icon)
        icon_label.setFont(QFont("Segoe UI Emoji", 28))
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setFixedWidth(56)
        left_col.addWidget(icon_label)

        cat_label = QLabel(m.category.upper())
        cat_label.setStyleSheet(f"""
            color: {m.color}; font-size: 9px; font-weight: 700;
            letter-spacing: 1px; padding: 2px 6px;
            background-color: {m.color}18; border-radius: 4px;
        """)
        cat_label.setAlignment(Qt.AlignCenter)
        left_col.addWidget(cat_label)

        layout.addLayout(left_col)

        # ── Centre: Name, description, details ────────────────────
        centre_col = QVBoxLayout()
        centre_col.setSpacing(3)
        centre_col.setAlignment(Qt.AlignTop)

        name_label = QLabel(m.name)
        name_label.setFont(QFont("Inter", 14, QFont.Bold))
        name_label.setStyleSheet(f"color: {THEME['text_primary']};")
        centre_col.addWidget(name_label)

        desc_label = QLabel(m.description)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet(f"color: {THEME['text_secondary']}; font-size: 11px;")
        centre_col.addWidget(desc_label)

        # Meta row: size + VRAM
        meta_parts = []
        if m.size_gb > 0:
            meta_parts.append(f"Size: {m.size_gb:.1f} GB")
        if m.vram_requirement_gb > 0:
            meta_parts.append(f"VRAM: {m.vram_requirement_gb:.0f} GB")
        if meta_parts:
            meta_label = QLabel("  |  ".join(meta_parts))
            meta_label.setStyleSheet(f"color: {THEME['text_muted']}; font-size: 10px;")
            centre_col.addWidget(meta_label)

        # Variant chips
        if m.variants:
            variant_text = "Variants: " + ", ".join(v.name for v in m.variants)
            variant_label = QLabel(variant_text)
            variant_label.setStyleSheet(f"color: {THEME['text_muted']}; font-size: 10px;")
            variant_label.setWordWrap(True)
            centre_col.addWidget(variant_label)

        centre_col.addStretch()
        layout.addLayout(centre_col, stretch=3)

        # ── Right: Status + Action ────────────────────────────────
        right_col = QVBoxLayout()
        right_col.setSpacing(6)
        right_col.setAlignment(Qt.AlignTop | Qt.AlignRight)

        # Status badge
        self.status_badge = QLabel()
        self._update_status_badge()
        self.status_badge.setAlignment(Qt.AlignCenter)
        right_col.addWidget(self.status_badge)

        # Circular progress (hidden by default)
        self.progress_widget = CircularProgressWidget()
        self.progress_widget.hide()
        right_col.addWidget(self.progress_widget, alignment=Qt.AlignCenter)

        # Action button
        self.action_btn = QPushButton()
        self.action_btn.setMinimumWidth(100)
        self.action_btn.setMinimumHeight(34)
        self._update_action_button()
        self.action_btn.clicked.connect(self._on_action)
        right_col.addWidget(self.action_btn)

        right_col.addStretch()
        layout.addLayout(right_col)

        # Apply card style
        self._apply_style()

    def _apply_style(self):
        m = self.model
        self.setStyleSheet(f"""
            QFrame#modelCard {{
                background-color: {THEME['bg_card']};
                border: 1px solid {THEME['border_default']};
                border-left: 3px solid {m.color};
                border-radius: 10px;
                padding: 2px;
            }}
            QFrame#modelCard:hover {{
                background-color: {THEME['bg_hover']};
                border-color: {m.color};
            }}
        """)

    def _update_status_badge(self):
        m = self.model
        status_map = {
            ModelStatus.INSTALLED:     ("Installed", THEME['accent_green'], f"{THEME['accent_green']}20"),
            ModelStatus.NOT_INSTALLED: ("Not Installed", THEME['text_muted'], f"{THEME['text_muted']}15"),
            ModelStatus.INSTALLING:    ("Installing…", THEME['accent_primary'], f"{THEME['accent_primary']}20"),
            ModelStatus.ERROR:         ("Error", THEME['accent_red'], f"{THEME['accent_red']}20"),
        }
        text, fg, bg = status_map.get(m.status, ("Unknown", THEME['text_muted'], THEME['bg_card']))
        self.status_badge.setText(text)
        self.status_badge.setStyleSheet(f"""
            color: {fg}; background-color: {bg};
            font-size: 10px; font-weight: 600;
            padding: 3px 10px; border-radius: 10px;
        """)

    def _update_action_button(self):
        m = self.model
        if m.status == ModelStatus.INSTALLED:
            self.action_btn.setText("Installed")
            self.action_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {THEME['accent_green']}20;
                    color: {THEME['accent_green']};
                    border: 1px solid {THEME['accent_green']}40;
                    border-radius: 8px;
                    padding: 6px 14px;
                    font-weight: 600; font-size: 11px;
                }}
                QPushButton:hover {{
                    background-color: {THEME['accent_green']}30;
                }}
            """)
            self.action_btn.setEnabled(False)
        elif m.status == ModelStatus.INSTALLING:
            self.action_btn.setText("Cancel")
            self.action_btn.setStyleSheet(STYLES['button_danger'])
            self.action_btn.setEnabled(True)
        elif m.status == ModelStatus.ERROR:
            self.action_btn.setText("Retry")
            self.action_btn.setStyleSheet(STYLES['button_primary'])
            self.action_btn.setEnabled(True)
        else:
            # NOT_INSTALLED
            self.action_btn.setText("Install")
            self.action_btn.setStyleSheet(f"""
                QPushButton {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 {THEME['accent_primary']}, stop:1 {m.color});
                    color: #fff; border: none; border-radius: 8px;
                    padding: 6px 14px; font-weight: 600; font-size: 11px;
                }}
                QPushButton:hover {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 {THEME['accent_hover']}, stop:1 {m.color});
                }}
            """)
            self.action_btn.setEnabled(m.install_method != InstallMethod.BUILT_IN)

    def update_status(self, model: ModelInfo):
        """Refresh the card from updated model info."""
        self.model = model
        self._update_status_badge()
        self._update_action_button()
        if model.status == ModelStatus.INSTALLING:
            self.progress_widget.show()
            self.progress_widget.set_progress(model.progress, model.color)
        else:
            self.progress_widget.hide()

    def set_progress(self, value: int):
        self.progress_widget.show()
        self.progress_widget.set_progress(value, self.model.color)

    def _on_action(self):
        if self.model.status == ModelStatus.NOT_INSTALLED or self.model.status == ModelStatus.ERROR:
            self.install_requested.emit(self.model.id)
        elif self.model.status == ModelStatus.INSTALLING:
            # Cancel — handled by parent
            pass


# ════════════════════════════════════════════════════════════════════
#  MODEL HUB DIALOG
# ════════════════════════════════════════════════════════════════════

class ModelHubDialog(QDialog):
    """Full-featured model browser and installer."""

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("Model Hub — Download & Install AI Models")
        self.setMinimumSize(900, 700)
        self.resize(1000, 750)

        self._cards: dict[str, ModelCardWidget] = {}
        self._download_thread: Optional[QThread] = None
        self._download_worker: Optional[ModelDownloadWorker] = None
        self._current_installing_id: Optional[str] = None

        self._build_ui()
        self._refresh_models()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Header ────────────────────────────────────────────────
        header = QFrame()
        header.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {THEME['accent_primary']}, stop:1 {THEME['accent_pink']});
                border: none;
            }}
        """)
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(24, 20, 24, 16)
        header_layout.setSpacing(4)

        title = QLabel("Model Hub")
        title.setFont(QFont("Inter", 20, QFont.Bold))
        title.setStyleSheet("color: #fff; background: transparent;")
        header_layout.addWidget(title)

        subtitle = QLabel("Browse, download, and install AI music models. Each model enables different generation capabilities.")
        subtitle.setStyleSheet("color: #ddd; font-size: 12px; background: transparent;")
        subtitle.setWordWrap(True)
        header_layout.addWidget(subtitle)

        layout.addWidget(header)

        # ── Stats Bar ─────────────────────────────────────────────
        self.stats_bar = QFrame()
        self.stats_bar.setStyleSheet(f"""
            QFrame {{
                background-color: {THEME['bg_secondary']};
                border-bottom: 1px solid {THEME['border_default']};
            }}
        """)
        stats_layout = QHBoxLayout(self.stats_bar)
        stats_layout.setContentsMargins(24, 8, 24, 8)

        self.stats_label = QLabel()
        self.stats_label.setStyleSheet(f"color: {THEME['text_muted']}; font-size: 11px;")
        stats_layout.addWidget(self.stats_label)
        stats_layout.addStretch()

        refresh_btn = QPushButton("Refresh All")
        refresh_btn.setStyleSheet(STYLES['button_secondary'])
        refresh_btn.clicked.connect(self._refresh_models)
        stats_layout.addWidget(refresh_btn)

        layout.addWidget(self.stats_bar)

        # ── Category Tabs ─────────────────────────────────────────
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # ── Install Console (bottom) ──────────────────────────────
        console_frame = QFrame()
        console_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {THEME['bg_primary']};
                border-top: 1px solid {THEME['border_default']};
            }}
        """)
        console_layout = QVBoxLayout(console_frame)
        console_layout.setContentsMargins(16, 8, 16, 8)
        console_layout.setSpacing(4)

        console_header = QHBoxLayout()
        console_title = QLabel("Install Console")
        console_title.setFont(QFont("Inter", 10, QFont.Bold))
        console_title.setStyleSheet(f"color: {THEME['text_muted']};")
        console_header.addWidget(console_title)
        console_header.addStretch()

        self.cancel_btn = QPushButton("Cancel Install")
        self.cancel_btn.setStyleSheet(STYLES['button_danger'])
        self.cancel_btn.hide()
        self.cancel_btn.clicked.connect(self._cancel_install)
        console_header.addWidget(self.cancel_btn)
        console_layout.addLayout(console_header)

        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setMaximumHeight(120)
        self.console_output.setStyleSheet(f"""
            QTextEdit {{
                background-color: {THEME['bg_primary']};
                color: {THEME['text_muted']};
                font-family: 'JetBrains Mono', 'Consolas', monospace;
                font-size: 10px;
                border: 1px solid {THEME['border_default']};
                border-radius: 6px;
                padding: 6px;
            }}
        """)
        console_layout.addWidget(self.console_output)

        layout.addWidget(console_frame)

    def _refresh_models(self):
        """Refresh all model statuses and rebuild cards."""
        models = refresh_all_statuses(self.config)
        categories = get_models_by_category(self.config)

        # Stats
        installed = sum(1 for m in models if m.status == ModelStatus.INSTALLED)
        total = len(models)
        self.stats_label.setText(
            f"{installed}/{total} models installed  |  "
            f"{total - installed} available to install"
        )

        # Clear and rebuild tabs
        self.tabs.clear()
        self._cards.clear()

        for cat_name, cat_models in sorted(categories.items()):
            tab = QWidget()
            tab_layout = QVBoxLayout(tab)
            tab_layout.setContentsMargins(12, 12, 12, 12)
            tab_layout.setSpacing(10)

            # Category description
            cat_desc = {
                "Vocal": "Models that generate singing and speech",
                "Generation": "Text-to-music and audio generation models",
                "Synth": "Synthesizers and DSP-based audio tools",
                "Utility": "Audio processing and analysis tools",
            }
            desc_label = QLabel(cat_desc.get(cat_name, ""))
            desc_label.setStyleSheet(f"color: {THEME['text_muted']}; font-size: 11px; padding: 4px 0;")
            tab_layout.addWidget(desc_label)

            for model in cat_models:
                card = ModelCardWidget(model)
                card.install_requested.connect(self._start_install)
                self._cards[model.id] = card
                tab_layout.addWidget(card)

            tab_layout.addStretch()

            # Count badge
            installed_in_cat = sum(1 for m in cat_models if m.status == ModelStatus.INSTALLED)
            tab_label = f"{cat_name} ({installed_in_cat}/{len(cat_models)})"
            self.tabs.addTab(tab, tab_label)

    def _start_install(self, model_id: str):
        """Begin installing a model."""
        if self._download_thread and self._download_thread.isRunning():
            QMessageBox.warning(
                self, "Busy",
                "Another installation is already running.\nPlease wait for it to finish or cancel it first."
            )
            return

        from model_manager import get_model_by_id
        model = get_model_by_id(model_id)
        if not model:
            return

        # Confirm
        desc = get_install_description(model, self.config)
        reply = QMessageBox.question(
            self, f"Install {model.name}?",
            f"Are you sure you want to install {model.name}?\n\n{desc}",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        # Update model status
        model.status = ModelStatus.INSTALLING
        model.progress = 0
        self._update_card(model_id)

        # Clear and show console
        self.console_output.clear()
        self.console_output.append(f"<b style='color:{THEME['accent_primary']}'>Installing {model.name}…</b>")
        self.cancel_btn.show()
        self._current_installing_id = model_id

        # Start worker
        self._download_thread = QThread()
        self._download_worker = ModelDownloadWorker(model, self.config)

        self._download_worker.moveToThread(self._download_thread)
        self._download_thread.started.connect(self._download_worker.run_install)

        self._download_worker.progress.connect(lambda v, mid=model_id: self._on_install_progress(mid, v))
        self._download_worker.log_line.connect(self._on_install_log)
        self._download_worker.finished.connect(self._on_install_finished)

        self._download_thread.start()

    def _cancel_install(self):
        if self._download_worker:
            self._download_worker.cancel()
            self.console_output.append(f"<span style='color:{THEME['accent_yellow']}'>Cancelling…</span>")

    @Slot(str, int)
    def _on_install_progress(self, model_id: str, value: int):
        if model_id in self._cards:
            self._cards[model_id].set_progress(value)

    @Slot(str)
    def _on_install_log(self, line: str):
        # Colour-code common pip output
        if "Successfully" in line:
            color = THEME['accent_green']
        elif "error" in line.lower() or "failed" in line.lower():
            color = THEME['accent_red']
        elif "Downloading" in line or "Collecting" in line:
            color = THEME['accent_primary']
        elif "Requirement already satisfied" in line:
            color = THEME['text_muted']
        else:
            color = THEME['text_secondary']

        self.console_output.append(f"<span style='color:{color}'>{line}</span>")
        # Auto-scroll
        self.console_output.verticalScrollBar().setValue(
            self.console_output.verticalScrollBar().maximum()
        )

    @Slot(bool, str)
    def _on_install_finished(self, success: bool, message: str):
        if self._download_thread:
            self._download_thread.quit()
            self._download_thread.wait()
            self._download_thread.deleteLater()
            self._download_worker.deleteLater()
            self._download_thread = None
            self._download_worker = None

        self.cancel_btn.hide()

        model_id = self._current_installing_id
        if model_id:
            from model_manager import get_model_by_id
            model = get_model_by_id(model_id)
            if model:
                model.status = ModelStatus.INSTALLED if success else ModelStatus.ERROR
                model.error_message = "" if success else message
                model.progress = 100 if success else 0
                self._update_card(model_id)

        if success:
            color = THEME['accent_green']
            self.console_output.append(f"<br><b style='color:{color}'>SUCCESS: {message}</b>")
        else:
            color = THEME['accent_red']
            self.console_output.append(f"<br><b style='color:{color}'>FAILED: {message}</b>")

        self._current_installing_id = None

        # Refresh all statuses
        QTimer.singleShot(500, self._refresh_models)

    def _update_card(self, model_id: str):
        if model_id in self._cards:
            from model_manager import get_model_by_id
            model = get_model_by_id(model_id)
            if model:
                self._cards[model_id].update_status(model)

    def closeEvent(self, event):
        if self._download_thread and self._download_thread.isRunning():
            reply = QMessageBox.question(
                self, "Installation Running",
                "An installation is still running. Cancel and close?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self._cancel_install()
                if self._download_thread:
                    self._download_thread.quit()
                    self._download_thread.wait()
            else:
                event.ignore()
                return
        event.accept()
