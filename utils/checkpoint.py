"""
Smart model checkpoint manager for FinSense.
Handles saving best models, periodic checkpoints, and cleanup.
"""

import logging
from pathlib import Path
import shutil
from typing import Optional, Dict, Any
import json


logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages model checkpoints with smart saving strategies.

    Features:
    - Save best performing model
    - Save periodic checkpoints
    - Keep only N best checkpoints
    - Automatic cleanup of old checkpoints
    - Metadata tracking (metrics, episode, etc.)
    """

    def __init__(self, checkpoint_dir='models', config=None):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir (str): Directory to save checkpoints
            config (dict): Checkpoint configuration
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if config is None:
            config = self._default_config()

        self.save_best = config.get('save_best', True)
        self.save_frequency = config.get('save_frequency', 10)
        self.max_keep = config.get('max_keep', 5)
        self.metric_name = config.get('metric_name', 'total_profit')
        self.mode = config.get('mode', 'max')  # 'max' or 'min'

        # Track best metric value
        self.best_metric = float('-inf') if self.mode == 'max' else float('inf')
        self.checkpoints = []  # List of (metric_value, filepath) tuples

        logger.info(f"Checkpoint manager initialized at {self.checkpoint_dir}")

    @staticmethod
    def _default_config():
        """Default checkpoint configuration."""
        return {
            'save_best': True,
            'save_frequency': 10,
            'max_keep': 5,
            'metric_name': 'total_profit',
            'mode': 'max'
        }

    def save_checkpoint(self, agent, episode, metrics, force=False):
        """
        Save checkpoint with smart strategy.

        Args:
            agent: Agent to save
            episode (int): Current episode number
            metrics (dict): Performance metrics
            force (bool): Force save regardless of strategy

        Returns:
            bool: True if checkpoint was saved
        """
        metric_value = metrics.get(self.metric_name, 0.0)

        # Determine if we should save
        should_save = force
        is_best = False

        # Check if this is the best model
        if self.save_best:
            if self._is_better(metric_value, self.best_metric):
                should_save = True
                is_best = True
                self.best_metric = metric_value
                logger.info(f"New best model! {self.metric_name}={metric_value:.4f}")

        # Check if periodic save
        if self.save_frequency > 0 and episode % self.save_frequency == 0:
            should_save = True

        if not should_save:
            return False

        # Save checkpoint
        if is_best:
            filepath = self.checkpoint_dir / 'best_model.pt'
            agent.save(filepath)
            logger.info(f"Saved best model to {filepath}")

            # Also save metadata
            self._save_metadata(filepath.with_suffix('.json'), episode, metrics, is_best=True)

        # Save periodic checkpoint
        if episode % self.save_frequency == 0:
            filepath = self.checkpoint_dir / f'model_ep{episode}.pt'
            agent.save(filepath)
            logger.info(f"Saved periodic checkpoint to {filepath}")

            # Save metadata
            self._save_metadata(filepath.with_suffix('.json'), episode, metrics)

            # Track checkpoint
            self.checkpoints.append((metric_value, filepath))

            # Cleanup old checkpoints
            self._cleanup_checkpoints()

        return True

    def _is_better(self, current, best):
        """Check if current metric is better than best."""
        if self.mode == 'max':
            return current > best
        else:
            return current < best

    def _save_metadata(self, filepath, episode, metrics, is_best=False):
        """Save checkpoint metadata."""
        metadata = {
            'episode': episode,
            'metrics': metrics,
            'is_best': is_best,
            'metric_name': self.metric_name,
            'metric_value': metrics.get(self.metric_name, 0.0)
        }

        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping only max_keep best."""
        if self.max_keep <= 0:
            return

        if len(self.checkpoints) <= self.max_keep:
            return

        # Sort checkpoints by metric
        if self.mode == 'max':
            self.checkpoints.sort(reverse=True, key=lambda x: x[0])
        else:
            self.checkpoints.sort(key=lambda x: x[0])

        # Remove worst checkpoints
        to_remove = self.checkpoints[self.max_keep:]
        self.checkpoints = self.checkpoints[:self.max_keep]

        for _, filepath in to_remove:
            if filepath.exists():
                filepath.unlink()
                logger.debug(f"Removed old checkpoint: {filepath}")

                # Remove metadata file
                metadata_file = filepath.with_suffix('.json')
                if metadata_file.exists():
                    metadata_file.unlink()

    def load_best_model(self, agent):
        """
        Load best saved model.

        Args:
            agent: Agent to load model into

        Returns:
            dict: Metadata of best model or None
        """
        best_path = self.checkpoint_dir / 'best_model.pt'

        if not best_path.exists():
            logger.warning("No best model found")
            return None

        agent.load(best_path)
        logger.info(f"Loaded best model from {best_path}")

        # Load metadata
        metadata_path = best_path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata

        return None

    def get_checkpoint_info(self):
        """Get information about all checkpoints."""
        info = {
            'best_metric': self.best_metric,
            'num_checkpoints': len(self.checkpoints),
            'checkpoints': []
        }

        for metric_value, filepath in self.checkpoints:
            metadata_path = filepath.with_suffix('.json')
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                info['checkpoints'].append({
                    'filepath': str(filepath),
                    'metric_value': metric_value,
                    'episode': metadata.get('episode', -1)
                })

        return info

    def get_latest_checkpoint(self):
        """
        Find the most recent checkpoint.

        Returns:
            tuple: (checkpoint_path, episode_number) or (None, 0) if no checkpoints
        """
        # Find all checkpoint files
        checkpoint_files = list(self.checkpoint_dir.glob('model_ep*.pt'))

        if not checkpoint_files:
            return None, 0

        # Find the one with highest episode number
        latest_checkpoint = None
        latest_episode = 0

        for ckpt_file in checkpoint_files:
            metadata_file = ckpt_file.with_suffix('.json')
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    episode = metadata.get('episode', 0)
                    if episode > latest_episode:
                        latest_episode = episode
                        latest_checkpoint = ckpt_file

        return latest_checkpoint, latest_episode

    def clear_checkpoints(self, keep_best=True):
        """
        Clear all checkpoints.

        Args:
            keep_best (bool): Keep the best model
        """
        for _, filepath in self.checkpoints:
            if filepath.exists():
                filepath.unlink()
                metadata_file = filepath.with_suffix('.json')
                if metadata_file.exists():
                    metadata_file.unlink()

        self.checkpoints = []

        if not keep_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            if best_path.exists():
                best_path.unlink()
            metadata_path = best_path.with_suffix('.json')
            if metadata_path.exists():
                metadata_path.unlink()

        logger.info("Checkpoints cleared")
