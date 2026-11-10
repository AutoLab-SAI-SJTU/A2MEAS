# Derived from LeRobot / OpenPI, Apache-2.0. See README.md.
from dataclasses import dataclass, field
from a2meas.robotwin.schema import PreTrainedConfig, ROBOTWIN_CAMERAS, ROBOTWIN_GRIPPERS
from a2meas.schema import NormalizationMode, FeatureType

@PreTrainedConfig.register_subclass('pi05')
@dataclass
class A2WLAConfig(PreTrainedConfig):
    dtype: str = 'bfloat16'
    rynnbrain_path: str = 'resources/assets/rynnbrain'
    lawam_config_path: str = ''
    lawam_vision_path: str = ''
    trajectory_query_count: int = 10
    lawam_primary_camera_key: str = 'observation.rgb.zed_link_camera_0'
    lawam_multi_view_camera_keys: list[str] = field(default_factory=lambda: ['observation.rgb.left_realsense_link_camera_0', 'observation.rgb.right_realsense_link_camera_0'])
    lawam_image_resolution: int = 256
    lawam_num_action_queries: int = 8
    lawam_num_prefix_tokens: int = 64
    chunk_size: int = 50
    max_state_dim: int = 32
    max_action_dim: int = 32
    num_inference_steps: int = 10
    min_period: float = 0.004
    max_period: float = 4.0
    short_term_memory_inference_delta_indices: list[int] = field(default_factory=lambda: [-30, 0])
    image_resolution: tuple[int, int] = (224, 224)
    normalization_mapping: dict[str, NormalizationMode] = field(default_factory=lambda: {'ACTION': 'QUANTILES', 'STATE': 'QUANTILES', 'VISUAL': 'IDENTITY'})
    flow_source_blend_alpha: float = 0.1
    default_speed_bin: int = 500
    gripper_action_indices: list[int] = field(default_factory=lambda: [6, 13])

    def __post_init__(self):
        super().__post_init__()
        if self.dtype not in {'bfloat16', 'float32'}:
            raise ValueError('Unsupported inference dtype')
        if (self.chunk_size, self.max_state_dim, self.max_action_dim, self.trajectory_query_count) != (50, 32, 32, 10):
            raise ValueError('RoboTwin release requires horizon 50, padded dimensions 32 and 10 trajectory queries')
        if self.short_term_memory_inference_delta_indices != [-30, 0]:
            raise ValueError('RoboTwin release requires two-frame history at offsets [-30, 0]')
        if self.num_inference_steps not in (10, 20):
            raise ValueError('Inference requires 10 or 20 denoising steps')
        if not 0 <= self.flow_source_blend_alpha <= 1:
            raise ValueError('flow_source_blend_alpha must be in [0, 1]')
        if self.default_speed_bin != 500:
            raise ValueError('RoboTwin release requires Speed 500')
        if not 0 < self.min_period <= self.max_period:
            raise ValueError('Time embedding periods must be positive and ordered')
        if len(self.image_resolution) != 2 or min(self.image_resolution) < 1 or self.image_resolution[0] != self.image_resolution[1]:
            raise ValueError('RoboTwin requires a positive square image resolution')

    def validate_features(self):
        if not self.input_features:
            raise ValueError('Missing non-empty input_features')
        if not self.output_features:
            raise ValueError('Missing non-empty output_features')
        for features, name, kind in ((self.input_features, 'observation.state', FeatureType.STATE), (self.output_features, 'action', FeatureType.ACTION)):
            feature = features.get(name)
            if feature is None or feature.type != kind or feature.shape != (14,):
                raise ValueError(f'RoboTwin {name} must be a 14D {kind.value} feature')
        for name in ROBOTWIN_CAMERAS:
            feature = self.input_features.get(name)
            if feature is None or feature.type != FeatureType.VISUAL or len(feature.shape) != 3 or (min(feature.shape) < 1) or (feature.shape[0] != 3 and feature.shape[-1] != 3):
                raise ValueError(f'Invalid RoboTwin RGB camera: {name}')
        if set(self.image_features) != set(ROBOTWIN_CAMERAS):
            raise ValueError('RoboTwin requires exactly the three configured policy cameras')
        if self.gripper_action_indices != ROBOTWIN_GRIPPERS:
            raise ValueError(f'RoboTwin gripper_action_indices must be {ROBOTWIN_GRIPPERS}')
