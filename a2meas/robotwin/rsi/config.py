"""Load the shared task registry before serving any episodes."""
import json
import os
from functools import lru_cache
from pathlib import Path

ACTORS = {'basket_inward_lift': ('basket_lift','BasketInwardLift'), 'bread_release_retreat': ('bread_retreat', 'BreadReleaseRetreat'), 'native_prefix': ('actor', 'Actor'), 'contact': ('contact', 'ContactSearch'), 'timing': ('timing', 'MotionTiming'),
          'proactive_grasp': ('grasp', 'ProactiveGrasp'),
          'can_release_cadence': ('can_release_cadence', 'CanReleaseCadence'),
          'placement': ('recovery', 'PlacementRecovery'), 'ranking': ('recovery', 'RankingRecovery'),
          'bottle': ('recovery', 'BottleRecovery'), 'mug': ('mug', 'MugRecovery')}
KINDS = set(ACTORS)


@lru_cache(maxsize=1)
def load_config(path=None):
    path = Path(path) if path is not None else Path(os.environ.get('ROBOTWIN_RSI_CONFIG', Path(__file__).resolve().parents[3] / 'configs/robotwin/rsi.json'))
    return read_config(path)


def read_config(path):
    path = Path(path)
    data = json.loads(path.read_text())
    if data.get('version') != 1 or not isinstance(data.get('tasks'), dict):
        raise ValueError('RSI config requires version=1 and a tasks object')
    if 'base_config' in data:
        raise ValueError('RSI configs must be complete; inherited profiles are not supported')
    defaults = data.get('defaults')
    if not isinstance(defaults, dict) or type(defaults.get('execution_chunk_size')) is not int or not 1 <= defaults['execution_chunk_size'] <= 50:
        raise ValueError('Invalid default execution_chunk_size')
    for task, settings in data['tasks'].items():
        if not isinstance(settings, dict):
            raise ValueError(f'Invalid RSI profile for {task}')
        if 'num_inference_steps' in settings:
            steps = settings['num_inference_steps']
            if type(steps) is not int or steps not in (10, 20):
                raise ValueError(f'num_inference_steps requires 10 or 20 for {task}')
        for flag in ('fixed_gaussian_noise', 'session_gaussian_noise', 'single_sample_inference', 'pan_only_role_fallback',
                     'observe_after_release_clearance', 'observe_before_close', 'consistent_hook_relocalization',
                     'retain_visible_ranking_slots', 'observe_at_release_end', 'fully_close_native_grasp',
                     'recent_visual_history'):
            if flag in settings and not isinstance(settings[flag], bool):
                raise ValueError(f'{flag} must be boolean for {task}')
        if settings.get('recent_visual_history') and not settings.get('single_sample_inference'):
            raise ValueError(f'recent_visual_history requires single_sample_inference for {task}')
        if settings.get('session_gaussian_noise'):
            if settings.get('fixed_gaussian_noise') or not settings.get('single_sample_inference'):
                raise ValueError(f'session_gaussian_noise requires single_sample_inference without fixed_gaussian_noise for {task}')
        if settings.get('single_sample_inference') and not (settings.get('fixed_gaussian_noise') or settings.get('session_gaussian_noise')):
            raise ValueError(f'single_sample_inference requires fixed_gaussian_noise or session_gaussian_noise for {task}')
        if settings.get('observe_before_close'):
            if settings.get('kind') not in ('placement', 'mug'):
                raise ValueError(f'observe_before_close requires placement or mug for {task}')
            if settings.get('kind') == 'mug' and settings.get('execution_chunk_size', data['defaults']['execution_chunk_size']) != 40:
                raise ValueError(f'observe_before_close requires the native40 mug window for {task}')
        if settings.get('consistent_hook_relocalization') and settings.get('kind') != 'mug':
            raise ValueError(f'consistent_hook_relocalization requires mug for {task}')
        if settings.get('fully_close_native_grasp') and settings.get('kind') != 'bottle':
            raise ValueError(f'fully_close_native_grasp requires bottle for {task}')
        if settings.get('retain_visible_ranking_slots') and settings.get('kind') != 'ranking':
            raise ValueError(f'retain_visible_ranking_slots requires ranking for {task}')
        if settings.get('observe_at_release_end'):
            if settings.get('kind') != 'can_release_cadence':
                raise ValueError(f'observe_at_release_end requires can_release_cadence for {task}')
            if settings.get('observe_after_release_clearance'):
                raise ValueError(f'observe_at_release_end conflicts with observe_after_release_clearance for {task}')
        if settings.get('kind') not in KINDS:
            raise ValueError(f'Unknown RSI kind for {task}: {settings.get("kind")}')
        if type(settings.get('max_steps')) is not int or settings['max_steps'] <= 0:
            raise ValueError(f'Invalid max_steps for {task}')
        chunk = settings.get('execution_chunk_size', data['defaults']['execution_chunk_size'])
        if type(chunk) is not int or not 1 <= chunk <= 50:
            raise ValueError(f'Invalid execution_chunk_size for {task}')
    return data


