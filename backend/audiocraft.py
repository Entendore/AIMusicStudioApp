"""AudioCraft / MusicGen backend — text-to-music generation."""
from __future__ import annotations

import logging
import numpy as np
from backend.base import MusicBackend

logger = logging.getLogger(__name__)


class Plugin(MusicBackend):
    """Meta AudioCraft (MusicGen) plugin: generates music from text descriptions.

    Supports small (300M), medium (1.5B), large (3.3B), and melody variants.
    Requires: ``pip install audiocraft``
    """

    def __init__(self):
        self.model = None
        self.model_name: str = "small"

    def load_model(self, config: dict):
        self.model_name = config.get("backend_settings", {}).get(
            "audiocraft", {}
        ).get("model_size", "small")

        try:
            from audiocraft.models import MusicGen
            device = config.get("hardware", {}).get("device", "cuda")
            self.model = MusicGen.get_pretrained(self.model_name, device=device)
            self.model.set_generation_params(duration=30)
            logger.info("AudioCraft '%s' loaded on %s", self.model_name, device)
        except ImportError:
            raise ImportError(
                "AudioCraft not found.  Install with:\n"
                "  pip install audiocraft"
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
        if self.model is None:
            raise RuntimeError("Model not loaded — call load_model() first")

        temperature = params.get("temperature", 1.0)
        top_k = params.get("top_k", 250)
        top_p = params.get("top_p", 0.0)
        guidance_scale = params.get("guidance_scale", 3.0)

        self.model.set_generation_params(
            duration=duration,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p if top_p > 0 else None,
            cfg_coef=guidance_scale,
        )

        logger.info(
            "AudioCraft generating: '%s' (%ds, temp=%.2f, top_k=%d)",
            prompt[:60], duration, temperature, top_k,
        )

        wav = self.model.generate([prompt], progress_callback=progress_callback)
        # wav shape: (batch, channels, samples)
        audio = wav[0, 0].cpu().numpy().astype(np.float32)

        if stop_check and stop_check():
            return None

        return {"audio": audio, "sample_rate": 32000}

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
            logger.info("AudioCraft model unloaded")

    @classmethod
    def get_settings_schema(cls):
        return [
            {
                "id": "model_size",
                "type": "dropdown",
                "label": "Model Size",
                "options": ["small", "medium", "large", "melody"],
                "default": "small",
                "category": "Model",
                "description": "small=300M, medium=1.5B, large=3.3B, melody=conditioned on melody",
            },
            {
                "id": "temperature",
                "type": "slider",
                "label": "Temperature",
                "min": 0.1,
                "max": 3.0,
                "step": 0.1,
                "default": 1.0,
                "category": "Generation",
                "description": "Controls randomness in generation",
            },
            {
                "id": "top_k",
                "type": "slider",
                "label": "Top-K",
                "min": 1,
                "max": 1000,
                "step": 1,
                "default": 250,
                "category": "Generation",
            },
            {
                "id": "top_p",
                "type": "slider",
                "label": "Top-P",
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
                "default": 0.0,
                "category": "Generation",
                "description": "0 = disabled (use top-k only)",
            },
            {
                "id": "guidance_scale",
                "type": "slider",
                "label": "Guidance",
                "min": 1.0,
                "max": 10.0,
                "step": 0.5,
                "default": 3.0,
                "category": "Generation",
            },
        ]

    @classmethod
    def get_prompt_schema(cls):
        return {
            "title": "Music Description",
            "description": "Describe the music you want — genre, instruments, mood, tempo",
            "placeholder": "80s synth pop with driving bassline, bright synthesizer leads, and a four-on-the-floor drum beat",
            "required": True,
        }
