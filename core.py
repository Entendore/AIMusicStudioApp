"""Core plugin management, audio utilities, and export support."""
from __future__ import annotations

import logging
import os
import pkgutil
import importlib
import json
from typing import Optional

import numpy as np
import soundfile as sf

from backend.base import MusicBackend

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class PluginManager:
    """Discovers, loads, and manages music generation backends."""

    def __init__(self):
        self.active_backend: Optional[MusicBackend] = None
        self._loaded_backends: dict[str, MusicBackend] = {}

    @staticmethod
    def discover_available_backends() -> list[str]:
        """Scan the backend package for valid Plugin classes."""
        available = []
        try:
            import backend as _pkg
            pkg_path = os.path.dirname(_pkg.__file__)
            for _importer, modname, ispkg in pkgutil.iter_modules([pkg_path]):
                if modname == 'base':
                    continue
                try:
                    module = importlib.import_module(f"backend.{modname}")
                    cls = getattr(module, "Plugin", None)
                    if cls is not None and issubclass(cls, MusicBackend) and cls is not MusicBackend:
                        available.append(modname)
                except Exception as exc:
                    logger.debug(f"Skipping backend '{modname}': {exc}")
        except ImportError:
            available = [
                "heartlib", "mathsynth", "audiocraft",
                "bark", "stable_audio", "riffusion", "ddsp", "demucs",
            ]
        return sorted(available)

    def load_plugin(self, plugin_name: str, config: dict) -> MusicBackend:
        """Instantiate and initialise a backend by name."""
        # Reuse if already loaded
        if plugin_name in self._loaded_backends:
            self.active_backend = self._loaded_backends[plugin_name]
            return self.active_backend

        module = importlib.import_module(f"backend.{plugin_name}")
        backend = module.Plugin()
        logger.info(f"Loading plugin: {plugin_name}…")
        backend.load_model(config)
        self._loaded_backends[plugin_name] = backend
        self.active_backend = backend
        return backend

    def switch_plugin(self, plugin_name: str, config: dict) -> MusicBackend:
        """Switch the active backend (does not unload others)."""
        if plugin_name in self._loaded_backends:
            self.active_backend = self._loaded_backends[plugin_name]
            return self.active_backend
        return self.load_plugin(plugin_name, config)

    def unload_all(self):
        """Unload every cached backend."""
        for name, backend in self._loaded_backends.items():
            try:
                backend.unload()
            except Exception:
                pass
        self._loaded_backends.clear()
        self.active_backend = None


# ── Audio Export ─────────────────────────────────────────────────────

def save_audio(audio_data: dict, filename: str, output_dir: str,
               fmt: str = "wav", subtype: str = "PCM_16") -> str:
    """Write audio data to disk and return the absolute path.

    Parameters
    ----------
    audio_data : dict
        ``{'audio': np.ndarray, 'sample_rate': int}``
    filename : str
        Base name without extension.
    output_dir : str
        Destination directory.
    fmt : str
        One of ``wav``, ``flac``, ``ogg``.
    subtype : str
        SoundFile subtype (e.g. ``PCM_16``, ``FLOAT``).
    """
    os.makedirs(output_dir, exist_ok=True)
    ext = fmt if fmt != "wav" else "wav"
    path = os.path.abspath(os.path.join(output_dir, f"{filename}.{ext}"))

    audio = audio_data['audio']
    sr = audio_data['sample_rate']

    # Ensure 2D for soundfile (samples, channels)
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)
    elif audio.ndim == 2 and audio.shape[0] < audio.shape[1]:
        audio = audio.T

    sf.write(path, audio, sr, subtype=subtype)
    logger.info(f"Saved audio → {path}")
    return path


def export_audio(audio_data: dict, filename: str, output_dir: str,
                 fmt: str = "wav") -> str:
    """Export audio with format-appropriate settings."""
    fmt_map = {
        'wav': ('wav', 'PCM_16'),
        'flac': ('flac', 'PCM_16'),
        'ogg': ('ogg', 'VORBIS'),
    }
    file_fmt, subtype = fmt_map.get(fmt, ('wav', 'PCM_16'))
    return save_audio(audio_data, filename, output_dir, file_fmt, subtype)


# ── Audio Effects (post-processing) ─────────────────────────────────

def apply_effects(audio: np.ndarray, sr: int, effects: dict) -> np.ndarray:
    """Apply a chain of simple DSP effects to an audio array.

    Supported effects (each key is a float 0–1 intensity unless noted):
        - ``normalize``  (bool)   — peak normalise to 0 dBFS
        - ``fade_in``    (float)  — fade-in duration in seconds
        - ``fade_out``   (float)  — fade-out duration in seconds
        - ``gain_db``    (float)  — gain in dB
        - ``reverb_mix`` (0–1)    — simple delay-based reverb mix
    """
    out = audio.copy().astype(np.float64)
    n = len(out)

    if effects.get('normalize', False):
        peak = np.max(np.abs(out))
        if peak > 0:
            out = out / peak * 0.95

    gain = effects.get('gain_db', 0.0)
    if gain != 0.0:
        out *= 10.0 ** (gain / 20.0)

    fade_in = effects.get('fade_in', 0.0)
    if fade_in > 0:
        samples = min(int(sr * fade_in), n // 2)
        out[:samples] *= np.linspace(0, 1, samples, dtype=np.float64)

    fade_out = effects.get('fade_out', 0.0)
    if fade_out > 0:
        samples = min(int(sr * fade_out), n // 2)
        out[-samples:] *= np.linspace(1, 0, samples, dtype=np.float64)

    reverb_mix = effects.get('reverb_mix', 0.0)
    if reverb_mix > 0:
        # Simple delay-based reverb (50ms delay, 3 taps)
        delay_samples = int(sr * 0.05)
        decay = 0.6
        reverb = np.zeros_like(out)
        for tap in range(1, 4):
            d = delay_samples * tap
            if d < n:
                reverb[d:] += out[:-d] * (decay ** tap)
        out = out * (1.0 - reverb_mix) + reverb * reverb_mix

    return out.astype(np.float32)


# ── Waveform Data ───────────────────────────────────────────────────

def compute_waveform_peaks(audio: np.ndarray, num_points: int = 800) -> np.ndarray:
    """Down-sample a 1-D audio array into *num_points* peak values for display."""
    if audio.ndim > 1:
        audio = audio[:, 0] if audio.shape[1] < audio.shape[0] else audio[0]
    chunk_size = max(len(audio) // num_points, 1)
    peaks = np.array([
        np.max(np.abs(audio[i:i + chunk_size]))
        for i in range(0, len(audio), chunk_size)
    ])
    # Trim or pad to exactly num_points
    if len(peaks) > num_points:
        peaks = peaks[:num_points]
    elif len(peaks) < num_points:
        peaks = np.pad(peaks, (0, num_points - len(peaks)))
    return peaks
