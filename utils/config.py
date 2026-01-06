"""
Configuration management for FinSense.
Loads and validates configuration from YAML files.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any


logger = logging.getLogger(__name__)


class Config:
    """
    Configuration manager for FinSense.

    Loads configuration from YAML files and provides easy access
    to configuration values.
    """

    def __init__(self, config_path='config.yaml'):
        """
        Initialize configuration.

        Args:
            config_path (str): Path to configuration YAML file

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid
        """
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config file: {e}")
            raise

        logger.info(f"Configuration loaded from {config_path}")

    def get(self, key, default=None):
        """
        Get configuration value by key.

        Args:
            key (str): Configuration key (supports dot notation, e.g., 'agent.gamma')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_section(self, section):
        """
        Get entire configuration section.

        Args:
            section (str): Section name (e.g., 'agent', 'training')

        Returns:
            dict: Section configuration or empty dict if not found
        """
        return self._config.get(section, {})

    def update(self, key, value):
        """
        Update configuration value.

        Args:
            key (str): Configuration key (supports dot notation)
            value: New value
        """
        keys = key.split('.')
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def save(self, path=None):
        """
        Save configuration to file.

        Args:
            path (str): Path to save config (uses original path if None)
        """
        save_path = Path(path) if path else self.config_path

        with open(save_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Configuration saved to {save_path}")

    def to_dict(self):
        """Return configuration as dictionary."""
        return self._config.copy()

    def __repr__(self):
        return f"Config({self.config_path})"


def load_config(config_path='config.yaml'):
    """
    Load configuration from YAML file.

    Args:
        config_path (str): Path to configuration file

    Returns:
        Config: Configuration object
    """
    return Config(config_path)
