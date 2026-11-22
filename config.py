"""Configuration loading, saving, environment setup, and preset management."""
import yaml
import os
import json
import copy
from datetime import datetime


# ── Default Configuration ────────────────────────────────────────────

DEFAULT_CONFIG = {
    'system': {
        'cache_dir': os.path.join(os.path.expanduser('~'), '.aimusicstudio', 'cache'),
        'ckpt_dir': os.path.join(os.path.expanduser('~'), '.aimusicstudio', 'checkpoints'),
        'heartlib_repo_dir': '',
        'theme': 'dark_pro',
        'window_geometry': None,
    },
    'hardware': {
        'device': 'auto',       # 'auto', 'cuda', 'cuda:0', 'cpu', 'mps'
        'precision': 'float16',
        'auto_vram_warning': True,
        'auto_release_gpu': True,
    },
    'plugins': {
        'active_plugin': 'mathsynth',
        'available_plugins': [
            'heartlib', 'mathsynth', 'audiocraft',
            'bark', 'stable_audio', 'riffusion', 'ddsp', 'demucs',
        ],
    },
    'settings': {
        'output_dir': os.path.join(os.path.expanduser('~'), 'AIMusicStudio_Output'),
        'export_format': 'wav',
        'auto_play': True,
        'auto_save': True,
        'history_limit': 100,
        'sample_rate': 44100,
    },
    'backend_settings': {},
    'presets': {},
    'history': [],
}


def load_config(config_path: str = None) -> dict:
    """Load config from YAML, falling back to defaults."""
    if config_path is None:
        config_path = os.path.join(
            os.path.expanduser('~'), '.aimusicstudio', 'config.yaml'
        )
    try:
        with open(config_path, 'r') as f:
            loaded = yaml.safe_load(f) or {}
        # Deep merge with defaults
        return _deep_merge(copy.deepcopy(DEFAULT_CONFIG), loaded)
    except FileNotFoundError:
        return copy.deepcopy(DEFAULT_CONFIG)


def save_config(config: dict, config_path: str = None):
    """Persist config to YAML."""
    if config_path is None:
        config_path = os.path.join(
            os.path.expanduser('~'), '.aimusicstudio', 'config.yaml'
        )
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def setup_environment(config: dict):
    """Set environment variables for HF / Torch cache directories."""
    cache_path = os.path.abspath(
        config.get('system', {}).get('cache_dir', DEFAULT_CONFIG['system']['cache_dir'])
    )
    os.makedirs(cache_path, exist_ok=True)

    os.environ['HF_HOME'] = cache_path
    os.environ['TORCH_HOME'] = cache_path
    os.environ['XDG_CACHE_HOME'] = cache_path


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, preserving nested keys."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


# ── Preset Management ────────────────────────────────────────────────

def save_preset(config: dict, preset_name: str, backend_name: str,
                prompt: str, duration: int, params: dict):
    """Save a generation preset for quick recall."""
    if 'presets' not in config:
        config['presets'] = {}
    config['presets'][preset_name] = {
        'backend': backend_name,
        'prompt': prompt,
        'duration': duration,
        'params': params,
        'created': datetime.now().isoformat(),
    }
    save_config(config)


def load_preset(config: dict, preset_name: str) -> dict | None:
    """Return a preset dict or None if not found."""
    return config.get('presets', {}).get(preset_name)


def delete_preset(config: dict, preset_name: str):
    """Remove a preset by name."""
    config.get('presets', {}).pop(preset_name, None)
    save_config(config)


def list_presets(config: dict) -> list[str]:
    """Return sorted list of preset names."""
    return sorted(config.get('presets', {}).keys())


# ── History Management ───────────────────────────────────────────────

def add_to_history(config: dict, filepath: str, backend: str,
                   prompt: str, duration: int):
    """Add a generation entry to the history log."""
    if 'history' not in config:
        config['history'] = []

    entry = {
        'filepath': filepath,
        'backend': backend,
        'prompt': prompt[:200],
        'duration': duration,
        'timestamp': datetime.now().isoformat(),
    }
    config['history'].insert(0, entry)

    # Trim to limit
    limit = config.get('settings', {}).get('history_limit', 100)
    config['history'] = config['history'][:limit]

    save_config(config)


def get_history(config: dict) -> list[dict]:
    """Return the generation history list."""
    return config.get('history', [])


def clear_history(config: dict):
    """Wipe the generation history."""
    config['history'] = []
    save_config(config)
