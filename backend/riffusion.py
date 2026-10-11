"""Riffusion backend — real-time music generation via spectrogram diffusion."""
from __future__ import annotations

import logging
import numpy as np
from backend.base import MusicBackend

logger = logging.getLogger(__name__)


class Plugin(MusicBackend):
    """Riffusion plugin: generates music clips from text using spectrogram diffusion.

    Based on Stable Diffusion fine-tuned on mel spectrograms.  Fast
    inference on consumer GPUs.
    Requires: ``pip install riffusion``
    """

    def __init__(self):
        self.pipeline = None

    def load_model(self, config: dict):
        try:
            from riffusion.cli import get_pipeline
            device = config.get("hardware", {}).get("device", "cuda")
            self.pipeline = get_pipeline()
            logger.info("Riffusion pipeline loaded")
        except ImportError:
            raise ImportError(
                "Riffusion not found.  Install with:\n"
                "  pip install riffusion"
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
        from riffusion.audio import wav_bytes_from_spectrogram_image
        import io

        denoising = params.get("denoising_strength", 0.75)
        guidance = params.get("guidance_scale", 7.0)
        seed = params.get("seed", -1)

        logger.info("Riffusion generating: '%s'", prompt[:60])

        # Riffusion generates ~5 second clips; we repeat/extend for longer durations
        params_dict = {
            "prompt": prompt,
            "num_inference_steps": 50,
            "guidance_scale": guidance,
            "denoising_strength": denoising,
        }
        if seed >= 0:
            params_dict["seed"] = seed

        image = self.pipeline.generate(**params_dict)
        wav_bytes = wav_bytes_from_spectrogram_image(image)

        # Decode wav bytes to numpy
        import soundfile as sf
        audio, sr = sf.read(io.BytesIO(wav_bytes))

        # Extend if needed
        target_samples = int(sr * duration)
        if len(audio) < target_samples:
            repeats = (target_samples // len(audio)) + 1
            audio = np.tile(audio, repeats)[:target_samples]

        if stop_check and stop_check():
            return None
        if progress_callback:
            progress_callback(100)

        return {"audio": audio.astype(np.float32), "sample_rate": sr}

    def unload(self):
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            logger.info("Riffusion pipeline unloaded")

    @classmethod
    def get_settings_schema(cls):
        return [
            {
                "id": "denoising_strength",
                "type": "slider",
                "label": "Denoising",
                "min": 0.1,
                "max": 1.0,
                "step": 0.05,
                "default": 0.75,
                "category": "Generation",
                "description": "How much to modify the initial noise",
            },
            {
                "id": "guidance_scale",
                "type": "slider",
                "label": "Guidance",
                "min": 1.0,
                "max": 15.0,
                "step": 0.5,
                "default": 7.0,
                "category": "Generation",
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
            },
        ]

    @classmethod
    def get_prompt_schema(cls):
        return {
            "title": "Music Prompt",
            "description": "Describe the music — Riffusion works best with genre + instrument descriptions",
            "placeholder": "funky bassline with jazz flute solo, 120 BPM",
            "required": True,
        }
