"""Stable Audio Open backend — text-to-audio via Stability AI."""
from __future__ import annotations

import logging
import numpy as np
from backend.base import MusicBackend

logger = logging.getLogger(__name__)


class Plugin(MusicBackend):
    """Stable Audio Open plugin: generates music and sound effects from text.

    Open-source model from Stability AI.  Good for instrumental music,
    sound effects, and ambient textures.
    Requires: ``pip install stable-audio-tools``
    """

    def __init__(self):
        self.model = None
        self.sample_rate = 44100

    def load_model(self, config: dict):
        try:
            import torch
            from stable_audio_tools import get_pretrained_model
            device = config.get("hardware", {}).get("device", "cuda")
            self.model, self.model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
            self.model = self.model.to(device)
            self.sample_rate = self.model_config.get("sample_rate", 44100)
            logger.info("Stable Audio Open loaded on %s", device)
        except ImportError:
            raise ImportError(
                "stable-audio-tools not found.  Install with:\n"
                "  pip install stable-audio-tools\n"
                "  See: https://github.com/Stability-AI/stable-audio-tools"
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
        import torch
        from stable_audio_tools import generate_from_prompt

        cfg_scale = params.get("cfg_scale", 7.0)
        steps = params.get("steps", 36)
        seed = params.get("seed", -1)

        if seed >= 0:
            torch.manual_seed(seed)

        conditioning = [{
            "prompt": prompt,
            "seconds_start": 0,
            "seconds_total": duration,
        }]

        output = generate_from_prompt(
            self.model,
            conditioning=conditioning,
            sample_size=int(self.sample_rate * duration),
            steps=steps,
            cfg_scale=cfg_scale,
            progress_callback=progress_callback,
        )

        if stop_check and stop_check():
            return None

        audio = output.squeeze().cpu().numpy().astype(np.float32)
        return {"audio": audio, "sample_rate": self.sample_rate}

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
            logger.info("Stable Audio model unloaded")

    @classmethod
    def get_settings_schema(cls):
        return [
            {
                "id": "cfg_scale",
                "type": "slider",
                "label": "CFG Scale",
                "min": 1.0,
                "max": 15.0,
                "step": 0.5,
                "default": 7.0,
                "category": "Generation",
                "description": "Classifier-free guidance scale",
            },
            {
                "id": "steps",
                "type": "slider",
                "label": "Steps",
                "min": 10,
                "max": 100,
                "step": 1,
                "default": 36,
                "category": "Generation",
                "description": "More steps = higher quality but slower",
            },
            {
                "id": "seed",
                "type": "spinbox",
                "label": "Seed",
                "min": -1,
                "max": 999999,
                "step": 1,
                "default": -1,
                "category": "Advanced",
                "description": "-1 = random",
            },
        ]

    @classmethod
    def get_prompt_schema(cls):
        return {
            "title": "Audio Description",
            "description": "Describe the audio: genre, instruments, mood, texture",
            "placeholder": "Ambient electronic pad with reverb, soft arpeggiated synth, 120 BPM",
            "required": True,
        }
