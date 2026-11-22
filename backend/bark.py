"""Bark backend — text-to-speech and text-to-music via Suno Bark."""
from __future__ import annotations

import logging
import numpy as np
from backend.base import MusicBackend

logger = logging.getLogger(__name__)


class Plugin(MusicBackend):
    """Suno Bark plugin: generates speech, singing, and music from text.

    Bark can produce vocal melodies, singing, and even instrumental
    passages when prompted with musical notation or lyrics.
    Requires: ``pip install git+https://github.com/suno-ai/bark.git``
    """

    def __init__(self):
        self.model = None

    def load_model(self, config: dict):
        try:
            from bark import SAMPLE_RATE, generate_audio, preload_models
            device = config.get("hardware", {}).get("device", "cuda")
            preload_models()
            self.model = True  # Bark uses module-level state
            self._sample_rate = SAMPLE_RATE
            logger.info("Bark models preloaded on %s", device)
        except ImportError:
            raise ImportError(
                "Bark not found.  Install with:\n"
                "  pip install git+https://github.com/suno-ai/bark.git"
            )

    def generate(
        self,
        prompt: str,
        duration: int,
        params: dict,
        *,
        progress_callback=None,
        stop_check=None,
    ):
        from bark import generate_audio as bark_generate

        voice = params.get("voice", "v2/en_speaker_6")
        text_temp = params.get("text_temperature", 0.7)
        waveform_temp = params.get("waveform_temperature", 0.7)

        logger.info("Bark generating: voice=%s, text_temp=%.2f", voice, text_temp)

        audio_array = bark_generate(
            prompt,
            history_prompt=voice,
            text_temp=text_temp,
            waveform_temp=waveform_temp,
        )

        if stop_check and stop_check():
            return None

        if progress_callback:
            progress_callback(100)

        return {"audio": audio_array.astype(np.float32), "sample_rate": self._sample_rate}

    def unload(self):
        self.model = None
        logger.info("Bark models released")

    @classmethod
    def get_settings_schema(cls):
        voices = [f"v2/en_speaker_{i}" for i in range(10)]
        return [
            {
                "id": "voice",
                "type": "dropdown",
                "label": "Voice",
                "options": voices,
                "default": "v2/en_speaker_6",
                "category": "Voice",
                "description": "Speaker voice preset",
            },
            {
                "id": "text_temperature",
                "type": "slider",
                "label": "Text Temp",
                "min": 0.1,
                "max": 1.5,
                "step": 0.05,
                "default": 0.7,
                "category": "Generation",
            },
            {
                "id": "waveform_temperature",
                "type": "slider",
                "label": "Wave Temp",
                "min": 0.1,
                "max": 1.5,
                "step": 0.05,
                "default": 0.7,
                "category": "Generation",
            },
        ]

    @classmethod
    def get_prompt_schema(cls):
        return {
            "title": "Text / Lyrics",
            "description": "Enter text for speech or singing. Use ♪ or la-la-la for musical output.",
            "placeholder": "♪ La la la, singing in the rain ♪",
            "required": True,
        }
