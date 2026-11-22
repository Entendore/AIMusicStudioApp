"""HeartMuLa (heartlib) backend — lyrics-to-song with vocals."""
from __future__ import annotations

import logging
import os
import sys
import numpy as np
from backend.base import MusicBackend

logger = logging.getLogger(__name__)


class Plugin(MusicBackend):
    """HeartMuLa plugin: generates songs with vocals from lyrics.

    Supports 3B and 7B model variants.  Optimised for GTX 1080 (8 GB VRAM)
    via 4-bit quantisation and gradient checkpointing.
    """

    def __init__(self):
        self.model = None
        self.model_name: str = "heartlib_3b"
        self.repo_dir: str = ""

    # ── Lifecycle ──────────────────────────────────────────────────

    def load_model(self, config: dict):
        self.repo_dir = config.get("system", {}).get("heartlib_repo_dir", "")
        self.model_name = config.get("backend_settings", {}).get(
            "heartlib", {}
        ).get("model_variant", "heartlib_3b")

        if not self.repo_dir or not os.path.isdir(self.repo_dir):
            raise FileNotFoundError(
                f"HeartMuLa repo not found at '{self.repo_dir}'.  "
                "Set the path in Settings → HeartMuLa Repo."
            )

        # Add repo to sys.path so we can import heartlib
        if self.repo_dir not in sys.path:
            sys.path.insert(0, self.repo_dir)

        try:
            from heartlib import HeartMuLaModel
            device = config.get("hardware", {}).get("device", "cuda")
            self.model = HeartMuLaModel.from_pretrained(
                self.model_name, device=device
            )
            logger.info("HeartMuLa model '%s' loaded on %s", self.model_name, device)
        except ImportError:
            raise ImportError(
                "Could not import heartlib.  Make sure you cloned the repo "
                "and installed its dependencies:\n"
                "  git clone https://github.com/HeartMuLa/heartlib.git\n"
                "  cd heartlib && pip install -e ."
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

        temperature = params.get("temperature", 0.9)
        top_k = params.get("top_k", 250)
        guidance_scale = params.get("guidance_scale", 3.0)
        seed = params.get("seed", -1)

        if seed >= 0:
            np.random.seed(seed)

        logger.info(
            "Generating with HeartMuLa: duration=%ds, temp=%.2f, top_k=%d, guidance=%.1f",
            duration, temperature, top_k, guidance_scale,
        )

        result = self.model.generate(
            lyrics=prompt,
            duration=duration,
            temperature=temperature,
            top_k=top_k,
            guidance_scale=guidance_scale,
            progress_callback=progress_callback,
            stop_check=stop_check,
        )
        return result

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
            logger.info("HeartMuLa model unloaded")

    # ── Schema ─────────────────────────────────────────────────────

    @classmethod
    def get_settings_schema(cls):
        return [
            {
                "id": "model_variant",
                "type": "dropdown",
                "label": "Model",
                "options": ["heartlib_3b", "heartlib_7b"],
                "default": "heartlib_3b",
                "category": "Model",
                "description": "3B is faster, 7B has higher quality",
            },
            {
                "id": "temperature",
                "type": "slider",
                "label": "Temperature",
                "min": 0.1,
                "max": 2.0,
                "step": 0.1,
                "default": 0.9,
                "category": "Generation",
                "description": "Higher = more random, Lower = more focused",
            },
            {
                "id": "top_k",
                "type": "slider",
                "label": "Top-K",
                "min": 1,
                "max": 500,
                "step": 1,
                "default": 250,
                "category": "Generation",
                "description": "Number of highest-probability tokens to sample from",
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
                "description": "Classifier-free guidance scale",
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
                "description": "-1 = random seed each time",
            },
        ]

    @classmethod
    def get_prompt_schema(cls):
        return {
            "title": "Lyrics",
            "description": "Enter lyrics with section markers like [Verse], [Chorus], [Bridge]",
            "placeholder": "[Verse]\nWalking through the midnight rain\n[Chorus]\nTake me higher, lift me up again",
            "required": True,
        }
