"""Demucs backend — source separation for existing audio files."""
from __future__ import annotations

import logging
import numpy as np
from backend.base import MusicBackend

logger = logging.getLogger(__name__)


class Plugin(MusicBackend):
    """Demucs plugin: separates audio into stems (vocals, drums, bass, other).

    Not a generator — instead, loads an existing audio file and
    separates it into individual instrument tracks.
    Requires: ``pip install demucs``
    """

    def __init__(self):
        self.model = None

    def load_model(self, config: dict):
        try:
            from demucs import pretrained
            variant = config.get("backend_settings", {}).get(
                "demucs", {}
            ).get("model_variant", "htdemucs")
            self.model = pretrained.get_model(variant)
            logger.info("Demucs model loaded")
        except ImportError:
            raise ImportError(
                "Demucs not found.  Install with:\n"
                "  pip install demucs"
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
        # prompt is treated as the input file path for Demucs
        import torchaudio
        from demucs.apply import apply_model

        stem = params.get("target_stem", "vocals")

        if not prompt or not os.path.isfile(prompt):
            raise FileNotFoundError(
                "For Demucs, the prompt field must contain a path to an audio file."
            )

        wav, sr = torchaudio.load(prompt)
        ref = wav.mean(0)
        wav_input = (wav - ref.mean()) / ref.std()

        sources = apply_model(self.model, wav_input[None], progress=progress_callback is not None)
        sources = sources * ref.std() + ref.mean()

        stem_idx = {"drums": 0, "bass": 1, "other": 2, "vocals": 3}.get(stem, 3)
        audio = sources[0, stem_idx].numpy().astype(np.float32)

        if stop_check and stop_check():
            return None

        return {"audio": audio, "sample_rate": sr}

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
            logger.info("Demucs model unloaded")

    @classmethod
    def get_settings_schema(cls):
        return [
            {
                "id": "model_variant",
                "type": "dropdown",
                "label": "Model",
                "options": ["htdemucs", "htdemucs_ft", "mdx", "mdx_extra"],
                "default": "htdemucs",
                "category": "Model",
            },
            {
                "id": "target_stem",
                "type": "dropdown",
                "label": "Extract Stem",
                "options": ["vocals", "drums", "bass", "other"],
                "default": "vocals",
                "category": "Separation",
            },
        ]

    @classmethod
    def get_prompt_schema(cls):
        return {
            "title": "Audio File Path",
            "description": "Path to the audio file to separate into stems",
            "placeholder": "/path/to/song.wav",
            "required": True,
        }
