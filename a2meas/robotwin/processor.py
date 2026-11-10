# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team.
"""Checkpoint quantiles and the released Speed500 inference prompt."""
from dataclasses import dataclass

from a2meas.processing.pipeline import (
    PolicyProcessorPipeline, ProcessorStep, TransitionKey,
    policy_action_to_transition, transition_to_policy_action,
)
from a2meas.processing.steps import DeviceProcessorStep, NormalizerProcessorStep, UnnormalizerProcessorStep


@dataclass
class PreparePrompt(ProcessorStep):
    default_speed_bin: int

    def __call__(self, transition):
        result = transition.copy()
        data = dict(transition[TransitionKey.COMPLEMENTARY_DATA])
        data['task'] = [
            f"Task: {task.strip().replace('_', ' ').replace(chr(10), ' ')}, Speed: {self.default_speed_bin}"
            for task in data['task']
        ]
        result[TransitionKey.COMPLEMENTARY_DATA] = data
        return result


def make_processors(config, pre_stats, post_stats):
    normalize = NormalizerProcessorStep(features=config.input_features, norm_map=config.normalization_mapping)
    denormalize = UnnormalizerProcessorStep(features=config.output_features, norm_map=config.normalization_mapping)
    normalize.load_state_dict({f'{key}.{name}': value for key, stats in pre_stats.items() for name, value in stats.items()})
    denormalize.load_state_dict({f'{key}.{name}': value for key, stats in post_stats.items() for name, value in stats.items()})
    return (
        PolicyProcessorPipeline(steps=[normalize, PreparePrompt(config.default_speed_bin), DeviceProcessorStep(device=config.device)]),
        PolicyProcessorPipeline(steps=[denormalize, DeviceProcessorStep(device='cpu')], to_transition=policy_action_to_transition, to_output=transition_to_policy_action),
    )
