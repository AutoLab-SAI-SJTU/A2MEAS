"""Load all inference tensors; reject missing or unknown checkpoint parameters."""
from safetensors import safe_open

_BINDINGS = {'model.trajectory_queries': 'model.trajectory_2d_tokens'}


def load_inference_weights(model, path):
    with safe_open(str(path), framework='pt', device='cpu') as checkpoint:
        keys = set(checkpoint.keys())
        aliases = checkpoint.metadata() or {}
        state, consumed, missing = {}, set(), []
        for name in model.state_dict():
            source = _BINDINGS.get(name, name)
            # Safetensors records tied embedding aliases instead of duplicating bytes.
            source = aliases.get(source, source)
            if source not in keys:
                missing.append(name)
            else:
                state[name] = checkpoint.get_tensor(source)
                consumed.add(source)
        unexpected = sorted(keys - consumed)
        if missing or unexpected:
            raise RuntimeError(f'Checkpoint mismatch: missing={missing}, unexpected={unexpected}')
        model.load_state_dict(state, strict=True)
