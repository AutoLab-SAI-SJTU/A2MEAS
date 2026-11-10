"""Validate the same policy configuration used by the evaluation server."""
from __future__ import annotations
import json
from pathlib import Path
from draccus.utils import ParsingError
from a2meas.robotwin.model.config import A2WLAConfig
from a2meas.robotwin.schema import ROBOTWIN_GRIPPERS

def _load_config(checkpoint: Path):
    for name in ('config.json', 'model.safetensors'):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(checkpoint / name)
    return json.loads((checkpoint / 'config.json').read_text())

def validate_checkpoint(checkpoint: Path, image_history_num_frames=None, execution_chunk_size=None, num_inference_steps=None):
    errors = []
    report = {'checkpoint': str(checkpoint), 'compatible': False, 'errors': errors, 'warnings': []}
    source = _load_config(checkpoint)
    try:
        config = A2WLAConfig.from_dict(source)
        config.validate_features()
    except (ValueError, TypeError, ParsingError) as exc:
        errors.append(str(exc))
        return report
    frames = 2
    if image_history_num_frames is not None and image_history_num_frames != frames:
        errors.append(f'Evaluation supplies {image_history_num_frames} STM frames; checkpoint requires {frames}')
    if execution_chunk_size is not None and (not 1 <= execution_chunk_size <= config.chunk_size):
        errors.append(f'Evaluation executes {execution_chunk_size} actions; checkpoint only predicts {config.chunk_size}')
    if num_inference_steps is not None and num_inference_steps != config.num_inference_steps:
        errors.append(f'Evaluation inference steps differ from checkpoint: {num_inference_steps} vs {config.num_inference_steps}')
    report.update(compatible=not errors, action_target_mode='absolute', flow_source_mode='blend', flow_source_state_num_frames=1, stm_enabled=True, stm_inference_frames=frames, chunk_size=config.chunk_size, resolved_gripper_indices=ROBOTWIN_GRIPPERS)
    return report
