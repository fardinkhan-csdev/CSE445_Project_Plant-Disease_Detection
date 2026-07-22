import os
import tempfile

from training.trainer import BaseTrainer


def test_checkpoint_paths_are_method_specific_and_timestamped():
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.experiment_name = 'lora'
    trainer.checkpoint_dir = tempfile.mkdtemp(prefix='checkpoint-test-', dir='.')

    primary_path, backup_path = trainer._build_checkpoint_paths('latest')

    assert os.path.basename(primary_path).startswith('lora_latest')
    assert os.path.basename(backup_path).startswith('lora_latest')
    assert primary_path != backup_path
    assert primary_path.endswith('.pth')
    assert backup_path.endswith('.pth')
