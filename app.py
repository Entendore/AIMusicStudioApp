#!/usr/bin/env python3
"""AI Music Studio Pro — Application entry point.

Configures paths, initialises environment, and launches the professional GUI.
"""
import os
import sys

# Resolve project root and add to sys.path
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import load_config, setup_environment
from core import PluginManager
from gui_main import launch_gui


def main():
    config = load_config()
    setup_environment(config)

    # Ensure output directory exists
    output_dir = config.get('settings', {}).get('output_dir', 'output')
    os.makedirs(output_dir, exist_ok=True)

    manager = PluginManager()
    launch_gui(manager, config)


if __name__ == "__main__":
    main()
