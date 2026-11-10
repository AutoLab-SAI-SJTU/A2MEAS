"""Native Gaussian sources for explicitly enabled task profiles."""


class FixedGaussianSource:
    def __init__(self, torch, action_dim):
        self.torch, self.action_dim, self.banks = torch, action_dim, {}

    def sample(self, native, shape, device, enabled):
        if not any(enabled):
            return native(shape, device)
        if len(enabled) != shape[0]:
            raise ValueError('Sampling flags must match the inference batch')
        torch = self.torch
        key = (tuple(shape[1:]), str(device))
        if key not in self.banks:
            generator = torch.Generator(device='cpu').manual_seed(0)
            bank = torch.randn((1,) + tuple(shape[1:]), generator=generator, dtype=torch.float32)
            bank[..., self.action_dim:] = 0.
            self.banks[key] = bank.to(device)
        bank = self.banks[key]
        if all(enabled):
            return bank.expand(shape).clone()
        output = native(shape, device)
        output[torch.tensor(enabled, dtype=torch.bool, device=device)] = bank[0]
        return output


class SessionGaussianSource:
    """Advance one native Gaussian stream per episode, independent of scheduling.

    Every episode starts at zero. The caller must hold the shared model lock.
    The native sampler retains responsibility for padding, dtype and scale.
    """
    def __init__(self, torch):
        self.torch = torch
        self.cpu_state = torch.Generator(device='cpu').manual_seed(0).get_state()
        self.cuda_state, self.device = None, None

    def sample(self, native, shape, device):
        torch = self.torch
        if shape[0] != 1:
            raise ValueError('Session Gaussian sampling requires one request')
        target = torch.device(device)
        gpu = torch.cuda.current_device() if target.type == 'cuda' and target.index is None else target.index
        key = (target.type, gpu)
        if self.device is not None and key != self.device:
            raise ValueError('An episode RNG stream cannot migrate between devices')
        self.device = key
        if target.type == 'cuda' and self.cuda_state is None:
            self.cuda_state = torch.Generator(device=target).manual_seed(0).get_state()
        # Preserve one normal global draw for requests outside this stream.
        native(shape, device)
        with torch.random.fork_rng(devices=[gpu] if target.type == 'cuda' else []):
            torch.random.set_rng_state(self.cpu_state)
            if target.type == 'cuda':
                torch.cuda.set_rng_state(self.cuda_state, device=gpu)
            output = native(shape, device)
            cpu = torch.random.get_rng_state().clone()
            cuda = torch.cuda.get_rng_state(gpu).clone() if target.type == 'cuda' else None
        self.cpu_state, self.cuda_state = cpu, cuda
        return output
