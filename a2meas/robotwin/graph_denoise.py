from collections import OrderedDict




class CudaReplayBackend:
    def __init__(self, torch, static_tensors=(), max_entries=128, max_bytes=24 * 1024 ** 3):
        self.torch = torch
        self.static_tensors = tuple(static_tensors)
        self.static_addresses = {tensor.data_ptr() for tensor in self.static_tensors}
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.entries = OrderedDict()
        self.bytes = 0
        self.streams = {}
        self.pools = {}
        self.graph_number = 0

    def _input_key(self, value):
        if isinstance(value, self.torch.Tensor):
            if value.device.type != 'cuda':
                raise ValueError('CUDA replay cannot capture CPU tensor inputs')
            if value.data_ptr() in self.static_addresses:
                return ('static', value.data_ptr())
            return ('tensor', tuple(value.shape), tuple(value.stride()), value.dtype, value.device)
        return ('scalar', type(value), str(value))

    def _allocate(self, value):
        shape = tuple(1 if stride == 0 else size for size, stride in zip(value.shape, value.stride()))
        storage = self.torch.empty_strided(shape, value.stride(), dtype=value.dtype, device=value.device)
        selection = tuple(slice(0, 1) if stride == 0 else slice(None) for stride in value.stride())
        storage.copy_(value[selection])
        return storage.expand(value.shape), storage, selection

    def __call__(self, model, example_inputs):
        from functorch.compile import make_boxed_func
        from torch._dynamo.backends.common import aot_autograd

        def compile_forward(graph, inputs):
            return make_boxed_func(self._compile_graph(graph, inputs))

        return aot_autograd(fw_compiler=compile_forward, inference_compiler=compile_forward)(model, example_inputs)

    def _compile_graph(self, model, example_inputs):
        graph_number = self.graph_number
        self.graph_number += 1

        def execute(*inputs):
            key = (graph_number, tuple(self._input_key(value) for value in inputs))
            if key in self.entries:
                entry = self.entries.pop(key)
                self.entries[key] = entry
            else:
                static_inputs = []
                copies = []
                size = 0
                device = None
                for index, value in enumerate(inputs):
                    if not isinstance(value, self.torch.Tensor):
                        static_inputs.append(value)
                    elif value.data_ptr() in self.static_addresses:
                        static_inputs.append(value)
                        device = value.device
                    else:
                        target, storage, selection = self._allocate(value)
                        static_inputs.append(target)
                        copies.append((index, storage, selection))
                        size += storage.untyped_storage().nbytes()
                        device = value.device
                if device is None:
                    raise ValueError('CUDA replay requires CUDA inputs')
                if size > self.max_bytes:
                    raise ValueError(f'One CUDA graph exceeds its input-cache budget: {size}')
                if device not in self.streams:
                    self.streams[device] = self.torch.cuda.Stream(device=device)
                    self.pools[device] = self.torch.cuda.graph_pool_handle()
                stream = self.streams[device]
                current = self.torch.cuda.current_stream(device)
                stream.wait_stream(current)
                with self.torch.cuda.stream(stream):
                    model(*static_inputs)
                stream.synchronize()
                graph = self.torch.cuda.CUDAGraph()
                with self.torch.cuda.graph(graph, pool=self.pools[device], stream=stream):
                    outputs = model(*static_inputs)
                current.wait_stream(stream)
                entry = (graph, outputs, copies, size)
                self.entries[key] = entry
                self.bytes += size
                while len(self.entries) > self.max_entries or self.bytes > self.max_bytes:
                    _, removed = self.entries.popitem(last=False)
                    self.bytes -= removed[3]
            graph, outputs, copies, _ = entry
            for index, storage, selection in copies:
                storage.copy_(inputs[index][selection])
            graph.replay()
            return self.torch.utils._pytree.tree_map(
                lambda value: value.clone() if isinstance(value, self.torch.Tensor) else value, outputs)
        return execute



class GraphDenoiser:
    def __init__(self, torch, function, dynamic=True):
        owner = getattr(function, '__self__', None)
        static_tensors = list(owner.parameters()) + list(owner.buffers()) if owner is not None else []
        self.backend = CudaReplayBackend(torch, static_tensors)
        self.compiled = torch.compile(function, backend=self.backend, dynamic=dynamic)

    def __call__(self, *arguments, **keywords):
        return self.compiled(*arguments, **keywords)



def configure_graph_denoiser(policy, torch):
    for module in policy.modules():
        original = getattr(module.forward, '_torchdynamo_orig_callable', None)
        if original is not None:
            module.forward = original
    torch._dynamo.config.disable = False
    torch._dynamo.config.recompile_limit = 128
    return GraphDenoiser(torch, policy.model.denoise_step)
