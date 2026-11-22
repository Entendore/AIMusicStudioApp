#!/usr/bin/env python3
"""Launch script for AI Music Studio Pro."""
import os
import sys

# Add project to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Ensure Qt can find EGL on headless systems
qt_lib = os.path.join(
    os.path.expanduser('~'), '.venv', 'lib', 'python3.12',
    'site-packages', 'PySide6', 'Qt', 'lib',
)
if os.path.isdir(qt_lib):
    os.environ.setdefault('LD_LIBRARY_PATH', qt_lib)

from config import load_config, setup_environment
from core import PluginManager
from gui_main import launch_gui

if __name__ == "__main__":
    config = load_config()
    setup_environment(config)

    output_dir = config.get('settings', {}).get('output_dir', 'output')
    os.makedirs(output_dir, exist_ok=True)

    manager = PluginManager()
    launch_gui(manager, config)
