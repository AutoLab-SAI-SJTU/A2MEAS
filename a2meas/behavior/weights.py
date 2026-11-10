"""Restore the pinned BEHAVIOR inference tensors from the released weight store."""

import base64
from contextlib import ExitStack
import ctypes
import json

import numpy as np

from a2meas.resources import download_files, verify_file


_zstd = ctypes.CDLL('libzstd.so.1')
_zstd.ZSTD_decompress.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
_zstd.ZSTD_decompress.restype = ctypes.c_size_t
_zstd.ZSTD_isError.argtypes = [ctypes.c_size_t]
_zstd.ZSTD_isError.restype = ctypes.c_uint


def decode_tensor(index, sources, tensor_id):
    entry = index['tensors'][tensor_id]
    source = sources[entry['file']]
    source.seek(entry['offset'])
    payload = source.read(entry['stored_bytes'])
    if len(payload) != entry['stored_bytes']:
        raise EOFError('Truncated BEHAVIOR weight shard')
    if entry['codec'] == 'raw':
        return payload
    if entry['codec'] not in ('zstd_shuffle', 'zstd_zigzag_shuffle'):
        raise ValueError(f"Unsupported tensor codec: {entry['codec']}")
    buffer = ctypes.create_string_buffer(entry['raw_bytes'])
    size = _zstd.ZSTD_decompress(buffer, len(buffer), payload, len(payload))
    if _zstd.ZSTD_isError(size) or size != entry['raw_bytes']:
        raise ValueError('Invalid compressed BEHAVIOR tensor')
    data = np.frombuffer(buffer.raw, dtype=np.uint8).reshape(entry['itemsize'], -1).T.copy().tobytes()
    if entry['reference'] is not None:
        reference = decode_tensor(index, sources, entry['reference'])
        if entry['codec'] == 'zstd_zigzag_shuffle':
            encoded = np.frombuffer(data, dtype='<u' + str(entry['itemsize']))
            delta = (encoded >> 1) ^ (np.zeros_like(encoded) - (encoded & 1))
            data = (delta + np.frombuffer(reference, dtype=encoded.dtype)).tobytes()
        else:
            data = np.bitwise_xor(np.frombuffer(data, dtype=np.uint8),
                                  np.frombuffer(reference, dtype=np.uint8)).tobytes()
    return data


def restore_models(destination, temporary, release, manifest, models):
    download_files(temporary, {'weights/index.json': release['files']['weights/index.json']}, release, None)
    index = json.loads((temporary / 'weights/index.json').read_text())
    if index['version'] != 3:
        raise ValueError('Unsupported BEHAVIOR weight store version')
    selected = {name: index['models'][name] for name in models}
    required = {tensor_id for model in selected.values() for tensor_id in model['tensors']}
    pending = list(required)
    while pending:
        reference = index['tensors'][pending.pop()]['reference']
        if reference is not None and reference not in required:
            required.add(reference)
            pending.append(reference)
    filenames = {f"weights/{index['tensors'][tensor_id]['file']}" for tensor_id in required}
    download_files(temporary, {name: release['files'][name] for name in sorted(filenames)}, release, None)
    with ExitStack() as stack:
        sources = {name.removeprefix('weights/'): stack.enter_context((temporary / name).open('rb'))
                   for name in filenames}
        for name, model in selected.items():
            target = destination / 'behavior/policies' / name / 'model.safetensors'
            partial = target.with_suffix('.partial')
            try:
                with partial.open('wb') as output:
                    output.write(base64.b64decode(model['header']))
                    for tensor_id in model['tensors']:
                        output.write(decode_tensor(index, sources, tensor_id))
                verify_file(partial, manifest[f'behavior/policies/{name}/model.safetensors'])
                partial.replace(target)
            finally:
                partial.unlink(missing_ok=True)
