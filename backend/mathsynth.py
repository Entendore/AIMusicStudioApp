"""MathSynth backend — pure-math waveform synthesizer (no GPU required)."""
from __future__ import annotations

import logging
import numpy as np
from backend.base import MusicBackend

logger = logging.getLogger(__name__)


class Plugin(MusicBackend):
    """Mathematical synthesizer that generates audio from waveform functions.

    Supports sine, sawtooth, square, FM synthesis, and noise types.
    No model download or GPU required — runs entirely on CPU.
    """

    def __init__(self):
        self.sample_rate = 44100

    def load_model(self, config: dict):
        self.sample_rate = config.get("backend_settings", {}).get(
            "mathsynth", {}
        ).get("sample_rate", 44100)
        logger.info("MathSynth ready (sample_rate=%d)", self.sample_rate)

    def generate(
        self,
        prompt: str,
        duration: int,
        params: dict,
        *,
        progress_callback=None,
        stop_check=None,
    ):
        sr = self.sample_rate
        n_samples = int(sr * duration)
        wave_type = params.get("wave_type", "sine")
        frequency = params.get("frequency", 440.0)
        amplitude = params.get("amplitude", 0.7)
        fm_depth = params.get("fm_depth", 0.0)
        fm_rate = params.get("fm_ratio", 2.0)
        noise_mix = params.get("noise_mix", 0.0)

        t = np.linspace(0, duration, n_samples, endpoint=False, dtype=np.float32)

        # Primary oscillator
        phase = 2.0 * np.pi * frequency * t

        if wave_type == "sine":
            audio = np.sin(phase)
        elif wave_type == "sawtooth":
            audio = 2.0 * (t * frequency - np.floor(0.5 + t * frequency))
        elif wave_type == "square":
            audio = np.sign(np.sin(phase))
        elif wave_type == "triangle":
            audio = 2.0 * np.abs(2.0 * (t * frequency - np.floor(t * frequency + 0.5))) - 1.0
        elif wave_type == "fm":
            modulator = np.sin(2.0 * np.pi * frequency * fm_rate * t)
            audio = np.sin(phase + fm_depth * modulator)
        else:
            audio = np.sin(phase)

        # Apply FM modulation if depth > 0 and not already FM type
        if fm_depth > 0 and wave_type != "fm":
            modulator = np.sin(2.0 * np.pi * frequency * fm_rate * t)
            audio = np.sin(phase + fm_depth * modulator)

        # Mix in noise
        if noise_mix > 0:
            if noise_mix <= 0.5:
                noise = np.random.randn(n_samples).astype(np.float32) * 0.5
            else:
                # Brownian noise
                noise = np.cumsum(np.random.randn(n_samples)).astype(np.float32)
                noise = noise / (np.max(np.abs(noise)) + 1e-9) * 0.5
            audio = audio * (1.0 - noise_mix) + noise * noise_mix

        # Amplitude envelope (fade in/out)
        fade_samples = min(int(sr * 0.05), n_samples // 4)
        if fade_samples > 0:
            fade_in = np.linspace(0, 1, fade_samples, dtype=np.float32)
            fade_out = np.linspace(1, 0, fade_samples, dtype=np.float32)
            audio[:fade_samples] *= fade_in
            audio[-fade_samples:] *= fade_out

        audio = (audio * amplitude).astype(np.float32)

        # Simulate progress
        if progress_callback:
            chunk = max(n_samples // 20, 1)
            for i in range(0, n_samples, chunk):
                if stop_check and stop_check():
                    return None
                progress_callback(int(100 * i / n_samples))
            progress_callback(100)

        logger.info("MathSynth: generated %.1fs %s @ %.0fHz", duration, wave_type, frequency)
        return {"audio": audio, "sample_rate": sr}

    def unload(self):
        logger.info("MathSynth unloaded")

    @classmethod
    def get_settings_schema(cls):
        return [
            {
                "id": "wave_type",
                "type": "dropdown",
                "label": "Waveform",
                "options": ["sine", "sawtooth", "square", "triangle", "fm"],
                "default": "sine",
                "category": "Oscillator",
                "description": "Primary oscillator waveform shape",
            },
            {
                "id": "frequency",
                "type": "slider",
                "label": "Frequency",
                "min": 20.0,
                "max": 2000.0,
                "step": 1.0,
                "default": 440.0,
                "category": "Oscillator",
                "description": "Fundamental frequency in Hz (A4 = 440 Hz)",
            },
            {
                "id": "amplitude",
                "type": "slider",
                "label": "Amplitude",
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
                "default": 0.7,
                "category": "Oscillator",
                "description": "Output volume level",
            },
            {
                "id": "fm_depth",
                "type": "slider",
                "label": "FM Depth",
                "min": 0.0,
                "max": 10.0,
                "step": 0.1,
                "default": 0.0,
                "category": "FM Synthesis",
                "description": "Frequency modulation depth (0 = no FM)",
            },
            {
                "id": "fm_ratio",
                "type": "slider",
                "label": "FM Ratio",
                "min": 0.5,
                "max": 8.0,
                "step": 0.5,
                "default": 2.0,
                "category": "FM Synthesis",
                "description": "Modulator-to-carrier frequency ratio",
            },
            {
                "id": "noise_mix",
                "type": "slider",
                "label": "Noise Mix",
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
                "default": 0.0,
                "category": "Noise",
                "description": "0 = clean tone, 0.5 = white noise blend, 1.0 = brown noise",
            },
        ]

    @classmethod
    def get_prompt_schema(cls):
        return {
            "title": "Note Label",
            "description": "Optional label for the generated tone",
            "placeholder": "e.g. A4 reference tone, Bass drop, Pad layer",
            "required": False,
        }
