"""Model Download Worker — runs pip install / git clone / HF download on a background thread."""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
from typing import Callable, Optional

from PySide6.QtCore import QObject, Signal, Slot

from model_manager import ModelInfo, InstallMethod, get_install_command

logger = logging.getLogger(__name__)


class ModelDownloadWorker(QObject):
    """Downloads and installs a model on a background QThread.

    Emits progress as 0–100, log lines as strings, and a final
    success(bool, str) signal.
    """
    progress = Signal(int)
    log_line = Signal(str)
    finished = Signal(bool, str)   # (success: bool, message: str)

    def __init__(self, model: ModelInfo, config: dict):
        super().__init__()
        self.model = model
        self.config = config
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    @Slot()
    def run_install(self):
        """Execute the model installation."""
        try:
            method = self.model.install_method

            if method == InstallMethod.BUILT_IN:
                self.finished.emit(True, "Already built-in — no installation needed.")
                return

            if method == InstallMethod.PIP:
                self._install_pip()
            elif method == InstallMethod.GIT_CLONE:
                self._install_git()
            elif method == InstallMethod.HF_DOWNLOAD:
                self._install_hf()
            else:
                self.finished.emit(False, f"Unknown install method: {method}")

        except Exception as e:
            logger.error(f"Install error: {e}")
            self.finished.emit(False, str(e))

    def _install_pip(self):
        """Run pip install in a subprocess, streaming output."""
        pkg = self.model.pip_package
        cmd = [sys.executable, "-m", "pip", "install", pkg]

        self.log_line.emit(f"$ {' '.join(cmd)}")
        self.progress.emit(5)

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Stream output
        total_lines = 100  # estimate
        lines_seen = 0
        for line in process.stdout:
            if self._is_cancelled:
                process.terminate()
                self.finished.emit(False, "Installation cancelled by user.")
                return

            line = line.rstrip()
            if line:
                self.log_line.emit(line)
                lines_seen += 1
                # Rough progress estimate
                pct = min(5 + int(85 * lines_seen / max(total_lines, 1)), 90)
                self.progress.emit(pct)

        process.wait()
        if process.returncode == 0:
            self.progress.emit(100)
            self.log_line.emit("Installation completed successfully!")
            self.finished.emit(True, f"{self.model.name} installed successfully.")
        else:
            self.progress.emit(0)
            self.log_line.emit(f"pip exited with code {process.returncode}")
            self.finished.emit(False, f"Installation failed (exit code {process.returncode}). Check the console for details.")

    def _install_git(self):
        """Clone a git repository."""
        git_url = self.model.git_url
        # Determine target directory
        target = self.config.get('system', {}).get(f'{self.model.id}_repo_dir', '')
        if not target:
            cache_dir = self.config.get('system', {}).get('cache_dir', '')
            target = os.path.join(cache_dir, self.model.id) if cache_dir else os.path.expanduser(f"~/.aimusicstudio/repos/{self.model.id}")

        cmd = ["git", "clone", git_url, target]
        self.log_line.emit(f"$ {' '.join(cmd)}")
        self.progress.emit(10)

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in process.stdout:
            if self._is_cancelled:
                process.terminate()
                self.finished.emit(False, "Clone cancelled by user.")
                return
            line = line.rstrip()
            if line:
                self.log_line.emit(line)
                self.progress.emit(min(90, 10 + 5))

        process.wait()
        if process.returncode == 0:
            # Save the repo dir in config
            self.config.setdefault('system', {})[f'{self.model.id}_repo_dir'] = target
            from config import save_config
            save_config(self.config)

            self.progress.emit(100)
            self.log_line.emit(f"Repository cloned to: {target}")
            self.finished.emit(True, f"{self.model.name} cloned successfully to {target}")
        else:
            self.progress.emit(0)
            self.finished.emit(False, f"Git clone failed (exit code {process.returncode})")

    def _install_hf(self):
        """Download model files from HuggingFace."""
        from huggingface_hub import snapshot_download

        repo_id = self.model.hf_repo
        if not repo_id and self.model.variants:
            repo_id = self.model.variants[0].hf_repo

        if not repo_id:
            self.finished.emit(False, "No HuggingFace repository specified.")
            return

        self.log_line.emit(f"Downloading from HuggingFace: {repo_id}")
        self.progress.emit(5)

        try:
            path = snapshot_download(
                repo_id=repo_id,
                progress_callback=self._hf_progress,
            )
            self.progress.emit(100)
            self.log_line.emit(f"Model downloaded to: {path}")
            self.finished.emit(True, f"{self.model.name} downloaded successfully.")
        except Exception as e:
            self.finished.emit(False, str(e))

    def _hf_progress(self, progress):
        """Callback from HuggingFace hub download."""
        if self._is_cancelled:
            raise KeyboardInterrupt("Download cancelled by user.")
        # progress might be a float or int depending on version
        try:
            pct = int(progress)
            self.progress.emit(min(pct, 99))
        except (TypeError, ValueError):
            pass
