"""Model management — registry, download tracking, install logic, and status queries."""
from __future__ import annotations

import logging
import os
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ── Enums ─────────────────────────────────────────────────────────────

class ModelStatus(Enum):
    """Installation status of a model."""
    NOT_INSTALLED = "not_installed"
    INSTALLING    = "installing"
    INSTALLED     = "installed"
    ERROR         = "error"
    UPDATE_AVAILABLE = "update_available"


class InstallMethod(Enum):
    """How a model can be installed."""
    PIP        = "pip"
    GIT_CLONE  = "git_clone"
    HF_DOWNLOAD = "hf_download"
    BUILT_IN   = "built_in"   # No installation needed


# ── Data Classes ──────────────────────────────────────────────────────

@dataclass
class ModelVariant:
    """A specific variant of a model (e.g., small vs large)."""
    id: str
    name: str
    size_gb: float
    hf_repo: str = ""
    filename: str = ""
    description: str = ""
    vram_requirement_gb: float = 0.0


@dataclass
class ModelInfo:
    """Complete metadata about an installable model/engine."""
    id: str                          # backend key (e.g. 'heartlib')
    name: str                        # Display name
    icon: str                        # Emoji icon
    category: str                    # Vocal, Generation, Synth, Utility
    description: str                 # Short one-liner
    details: str                     # Longer description
    color: str                       # Accent colour hex
    install_method: InstallMethod    = InstallMethod.BUILT_IN
    pip_package: str                 = ""     # e.g. "audiocraft"
    pip_version: str                 = ""     # Pin version e.g. ">=0.1.0"
    git_url: str                     = ""     # e.g. "https://github.com/HeartMuLa/heartlib.git"
    hf_repo: str                     = ""     # e.g. "facebook/musicgen-small"
    variants: list[ModelVariant]     = field(default_factory=list)
    status: ModelStatus              = ModelStatus.NOT_INSTALLED
    installed_version: str           = ""
    error_message: str               = ""
    progress: int                    = 0      # 0–100
    size_gb: float                   = 0.0
    vram_requirement_gb: float       = 0.0
    requirements_text: str           = ""
    homepage_url: str                = ""
    license: str                     = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d['install_method'] = self.install_method.value
        d['status'] = self.status.value
        return d


# ── Model Registry ───────────────────────────────────────────────────

MODEL_REGISTRY: list[ModelInfo] = [
    ModelInfo(
        id="mathsynth",
        name="MathSynth",
        icon="📊",
        category="Synth",
        description="Pure math waveform synthesizer — no AI model needed",
        details="Generates sine, sawtooth, square, triangle, FM synthesis, and noise waveforms. Runs entirely on CPU with zero setup. Perfect for testing the app or creating reference tones and bass drops.",
        color="#34d399",
        install_method=InstallMethod.BUILT_IN,
        size_gb=0.0,
        vram_requirement_gb=0.0,
        requirements_text="No additional requirements — works out of the box",
        license="MIT",
    ),
    ModelInfo(
        id="heartlib",
        name="HeartMuLa",
        icon="🎤",
        category="Vocal",
        description="Lyrics-to-song with AI vocals — the core engine",
        details="HeartMuLa generates full songs with singing vocals from lyrics. Supports 3B (fast) and 7B (quality) model variants. Optimized for GTX 1080 8GB VRAM via 4-bit quantization. Write lyrics with [Verse], [Chorus], [Bridge] markers.",
        color="#f472b6",
        install_method=InstallMethod.GIT_CLONE,
        git_url="https://github.com/HeartMuLa/heartlib.git",
        size_gb=12.0,
        vram_requirement_gb=6.0,
        variants=[
            ModelVariant("heartlib_3b", "3B (Fast)", 6.0,
                         hf_repo="HeartMuLa/heartlib-3b",
                         description="Faster generation, good quality"),
            ModelVariant("heartlib_7b", "7B (Quality)", 12.0,
                         hf_repo="HeartMuLa/heartlib-7b",
                         description="Higher quality vocals, slower"),
        ],
        requirements_text="Git clone + HuggingFace model download (~12 GB)",
        homepage_url="https://github.com/HeartMuLa/heartlib",
        license="Apache-2.0",
    ),
    ModelInfo(
        id="audiocraft",
        name="MusicGen (AudioCraft)",
        icon="🎵",
        category="Generation",
        description="Meta's text-to-music — describe music, get music",
        details="MusicGen generates instrumental music from text descriptions. Supports small (300M), medium (1.5B), large (3.3B), and melody-conditioned variants. Great for background music, beats, and instrumentals.",
        color="#7c5cfc",
        install_method=InstallMethod.PIP,
        pip_package="audiocraft",
        size_gb=3.3,
        vram_requirement_gb=4.0,
        variants=[
            ModelVariant("small", "Small (300M)", 0.6,
                         hf_repo="facebook/musicgen-small",
                         description="Fast, decent quality"),
            ModelVariant("medium", "Medium (1.5B)", 1.8,
                         hf_repo="facebook/musicgen-medium",
                         description="Good quality"),
            ModelVariant("large", "Large (3.3B)", 3.3,
                         hf_repo="facebook/musicgen-large",
                         description="Best quality, needs 16GB VRAM"),
            ModelVariant("melody", "Melody", 1.5,
                         hf_repo="facebook/musicgen-melody",
                         description="Conditioned on melody input"),
        ],
        requirements_text="pip install audiocraft + HuggingFace download (~1.8 GB for medium)",
        homepage_url="https://github.com/facebookresearch/audiocraft",
        license="MIT",
    ),
    ModelInfo(
        id="bark",
        name="Bark (Suno AI)",
        icon="🗣️",
        category="Vocal",
        description="Text-to-speech, singing, and music from text",
        details="Suno's Bark model generates realistic speech, singing, and even musical passages. Use ♪ or la-la-la for musical output. Supports multiple voice presets and languages. Can produce surprisingly musical results with the right prompts.",
        color="#fbbf24",
        install_method=InstallMethod.PIP,
        pip_package="git+https://github.com/suno-ai/bark.git",
        size_gb=2.5,
        vram_requirement_gb=4.0,
        requirements_text="pip install git+https://github.com/suno-ai/bark.git (~2.5 GB)",
        homepage_url="https://github.com/suno-ai/bark",
        license="MIT",
    ),
    ModelInfo(
        id="stable_audio",
        name="Stable Audio Open",
        icon="🔊",
        category="Generation",
        description="Stability AI's text-to-audio for music and SFX",
        details="Open-source model from Stability AI for generating instrumental music, sound effects, and ambient textures. Good at long-form ambient and electronic music. 44.1kHz output quality.",
        color="#60a5fa",
        install_method=InstallMethod.PIP,
        pip_package="stable-audio-tools",
        size_gb=3.2,
        vram_requirement_gb=6.0,
        variants=[
            ModelVariant("default", "Stable Audio Open 1.0", 3.2,
                         hf_repo="stabilityai/stable-audio-open-1.0",
                         description="Full quality model"),
        ],
        requirements_text="pip install stable-audio-tools + HuggingFace download (~3.2 GB)",
        homepage_url="https://github.com/Stability-AI/stable-audio-tools",
        license="StabilityAI Community",
    ),
    ModelInfo(
        id="riffusion",
        name="Riffusion",
        icon="🎸",
        category="Generation",
        description="Real-time music from text via spectrogram diffusion",
        details="Uses Stable Diffusion fine-tuned on mel spectrograms for fast music generation. Generates ~5 second clips that can be chained. Very fast inference — great for quick ideas and experimentation on consumer GPUs.",
        color="#f87171",
        install_method=InstallMethod.PIP,
        pip_package="riffusion",
        size_gb=4.0,
        vram_requirement_gb=4.0,
        requirements_text="pip install riffusion (~4 GB)",
        homepage_url="https://github.com/riffusion/riffusion",
        license="MIT",
    ),
    ModelInfo(
        id="ddsp",
        name="DDSP",
        icon="🎹",
        category="Synth",
        description="Differentiable DSP — realistic instrument timbre transfer",
        details="Google's DDSP (Differentiable Digital Signal Processing) enables realistic instrument synthesis and timbre transfer. Transform simple oscillator input into violin, flute, trumpet, and more using neural network-processed harmonic + noise models.",
        color="#7c5cfc",
        install_method=InstallMethod.PIP,
        pip_package="ddsp",
        size_gb=0.5,
        vram_requirement_gb=2.0,
        requirements_text="pip install ddsp (~0.5 GB)",
        homepage_url="https://github.com/magenta/ddsp",
        license="Apache-2.0",
    ),
    ModelInfo(
        id="demucs",
        name="Demucs",
        icon="🎛️",
        category="Utility",
        description="Source separation — split songs into vocal/drum/bass/other stems",
        details="Meta's Demucs separates any audio file into 4 stems: vocals, drums, bass, and other. Perfect for remixing, sampling, or extracting acapellas. Supports htdemucs (standard) and htdemucs_ft (fine-tuned) models.",
        color="#34d399",
        install_method=InstallMethod.PIP,
        pip_package="demucs",
        size_gb=1.5,
        vram_requirement_gb=2.0,
        variants=[
            ModelVariant("htdemucs", "HTDemucs (Standard)", 0.8,
                         description="Default separation model"),
            ModelVariant("htdemucs_ft", "HTDemucs FT", 1.5,
                         description="Fine-tuned, better quality"),
        ],
        requirements_text="pip install demucs + model download (~1.5 GB)",
        homepage_url="https://github.com/facebookresearch/demucs",
        license="MIT",
    ),
]


# ── Model Status Checking ─────────────────────────────────────────────

def check_model_status(model: ModelInfo, config: dict) -> ModelStatus:
    """Check whether a model's dependencies are installed."""
    if model.install_method == InstallMethod.BUILT_IN:
        return ModelStatus.INSTALLED

    if model.install_method == InstallMethod.PIP:
        return _check_pip_status(model)

    if model.install_method == InstallMethod.GIT_CLONE:
        return _check_git_status(model, config)

    if model.install_method == InstallMethod.HF_DOWNLOAD:
        return _check_hf_status(model, config)

    return ModelStatus.NOT_INSTALLED


def _check_pip_status(model: ModelInfo) -> ModelStatus:
    """Check if a pip package is importable."""
    pkg_name = model.pip_package.split("+")[-1].split("/")[-1].replace(".git", "")
    # Try common import name mappings
    import_map = {
        "audiocraft": "audiocraft",
        "bark": "bark",
        "stable-audio-tools": "stable_audio_tools",
        "riffusion": "riffusion",
        "ddsp": "ddsp",
        "demucs": "demucs",
    }
    import_name = import_map.get(pkg_name, pkg_name.replace("-", "_"))
    try:
        __import__(import_name)
        return ModelStatus.INSTALLED
    except ImportError:
        return ModelStatus.NOT_INSTALLED


def _check_git_status(model: ModelInfo, config: dict) -> ModelStatus:
    """Check if a git-cloned repo exists."""
    repo_dir = config.get('system', {}).get(f'{model.id}_repo_dir', '')
    if not repo_dir:
        repo_dir = config.get('system', {}).get('heartlib_repo_dir', '')
    if repo_dir and os.path.isdir(os.path.join(repo_dir, '.git')):
        return ModelStatus.INSTALLED
    return ModelStatus.NOT_INSTALLED


def _check_hf_status(model: ModelInfo, config: dict) -> ModelStatus:
    """Check if HuggingFace model files are cached locally."""
    hf_home = os.environ.get('HF_HOME', '')
    if not hf_home or not os.path.isdir(hf_home):
        return ModelStatus.NOT_INSTALLED
    # Check if any files from the repo are cached
    for variant in model.variants:
        if variant.hf_repo:
            cache_path = os.path.join(hf_home, "hub", f"models--{variant.hf_repo.replace('/', '--')}")
            if os.path.isdir(cache_path):
                return ModelStatus.INSTALLED
    if model.hf_repo:
        cache_path = os.path.join(hf_home, "hub", f"models--{model.hf_repo.replace('/', '--')}")
        if os.path.isdir(cache_path):
            return ModelStatus.INSTALLED
    return ModelStatus.NOT_INSTALLED


def refresh_all_statuses(config: dict) -> list[ModelInfo]:
    """Update and return all model statuses."""
    for model in MODEL_REGISTRY:
        model.status = check_model_status(model, config)
    return MODEL_REGISTRY


def get_model_by_id(model_id: str) -> Optional[ModelInfo]:
    """Look up a ModelInfo by its id."""
    for m in MODEL_REGISTRY:
        if m.id == model_id:
            return m
    return None


def get_installed_models(config: dict) -> list[ModelInfo]:
    """Return only the models that are currently installed."""
    refresh_all_statuses(config)
    return [m for m in MODEL_REGISTRY if m.status == ModelStatus.INSTALLED]


def get_models_by_category(config: dict) -> dict[str, list[ModelInfo]]:
    """Group models by category."""
    refresh_all_statuses(config)
    cats: dict[str, list[ModelInfo]] = {}
    for m in MODEL_REGISTRY:
        cats.setdefault(m.category, []).append(m)
    return cats


# ── Install Commands ──────────────────────────────────────────────────

def get_install_command(model: ModelInfo) -> list[str]:
    """Return the shell command to install a model's pip package."""
    if model.install_method == InstallMethod.PIP:
        return [sys.executable, "-m", "pip", "install", model.pip_package]
    if model.install_method == InstallMethod.GIT_CLONE:
        return ["git", "clone", model.git_url]
    return []


def get_install_description(model: ModelInfo, config: dict) -> str:
    """Return a human-readable description of what will happen during install."""
    if model.install_method == InstallMethod.BUILT_IN:
        return "No installation needed — this engine is built in."
    if model.install_method == InstallMethod.PIP:
        return f"Will run:  pip install {model.pip_package}\n\nThis will download ~{model.size_gb:.1f} GB of model files to your HuggingFace cache."
    if model.install_method == InstallMethod.GIT_CLONE:
        repo_dir = config.get('system', {}).get(f'{model.id}_repo_dir', '')
        target = repo_dir or f"~/aimusicstudio/{model.id}"
        return f"Will clone:  git clone {model.git_url}\n\nTarget: {target}\n\nAfter cloning, model weights (~{model.size_gb:.1f} GB) will be downloaded on first use."
    if model.install_method == InstallMethod.HF_DOWNLOAD:
        return f"Will download model from HuggingFace:\n  {model.hf_repo}\n\nSize: ~{model.size_gb:.1f} GB"
    return "Unknown installation method."


# ── VRAM Compatibility ────────────────────────────────────────────────

def check_vram_compatibility(model: ModelInfo, config: dict) -> dict:
    """Check if the system GPU has enough VRAM for a model.

    Returns a dict with:
        - 'compatible': bool
        - 'device': str — recommended device
        - 'reason': str — human-readable explanation
        - 'headroom_gb': float — free VRAM after model load
    """
    try:
        from gpu_manager import get_gpu_manager
        gm = get_gpu_manager()
        rec = gm.recommend_device(
            vram_needed_gb=model.vram_requirement_gb,
            model_name=model.name,
        )
        return {
            'compatible': rec.vram_sufficient,
            'device': rec.device,
            'reason': rec.reason,
            'headroom_gb': rec.vram_headroom_mb / 1024.0,
        }
    except Exception as e:
        return {
            'compatible': True,  # Assume compatible if detection fails
            'device': config.get('hardware', {}).get('device', 'auto'),
            'reason': f"Could not check GPU: {e}",
            'headroom_gb': 0.0,
        }


def get_compatible_models(config: dict) -> list[ModelInfo]:
    """Return models that can run on the current GPU (or CPU-only models)."""
    models = refresh_all_statuses(config)
    compatible = []
    for m in models:
        if m.vram_requirement_gb <= 0:
            compatible.append(m)  # CPU-only models always compatible
        else:
            check = check_vram_compatibility(m, config)
            if check['compatible'] or check['device'] != 'cpu':
                compatible.append(m)  # GPU models that fit or may fit with offloading
    return compatible
