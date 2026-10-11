"""Background worker threads, Qt logging bridge, and export/effects workers."""
from __future__ import annotations

import datetime
import logging
import os
from typing import Optional

from PySide6.QtCore import QObject, Signal, Slot


# ── Logging Bridge ──────────────────────────────────────────────────

class LogEmitter(QObject):
    """Thread-safe bridge: emits log records as Qt signals."""
    log_message = Signal(str)


class QtLogHandler(logging.Handler):
    """Redirects Python log records into a LogEmitter signal."""

    def __init__(self, emitter: LogEmitter):
        super().__init__()
        self.emitter = emitter

    def emit(self, record):
        msg = self.format(record)
        self.emitter.log_message.emit(msg)


# ── Generation Worker ──────────────────────────────────────────────

class Worker(QObject):
    """Runs audio generation on a background QThread."""
    finished = Signal()
    error = Signal(str)
    success = Signal(dict, str)
    progress = Signal(int)

    def __init__(self, backend, prompt, duration, params, config):
        super().__init__()
        self.backend = backend
        self.prompt = prompt
        self.duration = duration
        self.params = params
        self.config = config
        self._is_cancelled = False

    def stop(self):
        """Request cancellation of the running generation."""
        self._is_cancelled = True

    @Slot()
    def run_generation(self):
        """Execute the generation loop; emit signals for progress/result."""
        try:
            result = self.backend.generate(
                self.prompt,
                self.duration,
                self.params,
                progress_callback=self.progress.emit,
                stop_check=lambda: self._is_cancelled,
            )

            if result is not None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_prompt = "".join(
                    c if c.isalnum() else "_" for c in self.prompt
                )[:20]
                filename = f"{timestamp}_{safe_prompt}"
                self.success.emit(result, filename)

        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()


# ── Export Worker ───────────────────────────────────────────────────

class ExportWorker(QObject):
    """Exports generated audio to a different format on a background thread."""
    finished = Signal()
    error = Signal(str)
    success = Signal(str)   # emitted with the exported file path
    progress = Signal(int)

    def __init__(self, audio_data: dict, filename: str,
                 output_dir: str, fmt: str = "wav"):
        super().__init__()
        self.audio_data = audio_data
        self.filename = filename
        self.output_dir = output_dir
        self.fmt = fmt

    @Slot()
    def run_export(self):
        try:
            from core import export_audio
            self.progress.emit(30)
            path = export_audio(
                self.audio_data, self.filename,
                self.output_dir, self.fmt,
            )
            self.progress.emit(100)
            self.success.emit(path)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()


# ── Effects Worker ──────────────────────────────────────────────────

class EffectsWorker(QObject):
    """Applies post-processing effects on a background thread."""
    finished = Signal()
    error = Signal(str)
    success = Signal(dict)  # emits new audio_data dict
    progress = Signal(int)

    def __init__(self, audio_data: dict, effects: dict):
        super().__init__()
        self.audio_data = audio_data
        self.effects = effects

    @Slot()
    def run_effects(self):
        try:
            import numpy as np
            from core import apply_effects
            self.progress.emit(20)

            audio = self.audio_data['audio']
            sr = self.audio_data['sample_rate']
            processed = apply_effects(audio, sr, self.effects)

            self.progress.emit(80)
            result = {
                'audio': processed,
                'sample_rate': sr,
            }
            self.success.emit(result)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()
