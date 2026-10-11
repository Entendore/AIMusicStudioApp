"""Abstract base class that every music-generation backend must implement."""
from __future__ import annotations

import abc
from typing import Callable, Optional


class MusicBackend(abc.ABC):
    """Interface for all AI music generation backends.

    Subclasses **must** override:
        * ``load_model``   — load / prepare the model weights
        * ``generate``     — produce audio from a prompt
        * ``unload``       — free GPU / CPU resources

    Subclasses **may** override:
        * ``get_settings_schema`` — declare adjustable parameters
        * ``get_prompt_schema``   — customise the prompt input
    """

    # ── Lifecycle ──────────────────────────────────────────────────

    @abc.abstractmethod
    def load_model(self, config: dict):
        """Load model checkpoints & move to device.  Called once at startup."""

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        duration: int,
        params: dict,
        *,
        progress_callback: Optional[Callable[[int], None]] = None,
        stop_check: Optional[Callable[[], bool]] = None,
    ) -> dict:
        """Generate audio and return ``{'audio': np.ndarray, 'sample_rate': int}``.

        Parameters
        ----------
        prompt : str
            Text prompt or lyrics (backend-specific).
        duration : int
            Target duration in seconds.
        params : dict
            Key-value pairs from the settings widgets.
        progress_callback : callable, optional
            ``progress_callback(0–100)`` to update the UI progress bar.
        stop_check : callable, optional
            Return ``True`` to signal cancellation.
        """

    def unload(self):
        """Release model resources.  Default is a no-op."""

    # ── Schema hooks ───────────────────────────────────────────────

    @classmethod
    def get_settings_schema(cls) -> list[dict]:
        """Return a list of setting definitions for the UI.

        Each item is a dict with at least ``id``, ``type``, ``label``.
        Supported types: ``slider``, ``dropdown``, ``text``, ``spinbox``.
        """
        return []

    @classmethod
    def get_prompt_schema(cls) -> dict:
        """Return metadata about the prompt input.

        Keys: ``title``, ``description``, ``placeholder``, ``required``.
        """
        return {
            "title": "Prompt",
            "description": "Describe the music you want to generate",
            "placeholder": "Enter a description…",
            "required": True,
        }
