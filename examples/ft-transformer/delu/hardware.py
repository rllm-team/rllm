"""Tools related to devices, memory, etc."""

from typing import Any, Dict

from ._utils import deprecated
from .cuda import free_memory as free_memory_original

try:
    import pynvml
except ImportError:
    pynvml = None
if pynvml is not None:
    from pynvml import NVMLError_LibraryNotFound


def _to_str(x):
    if isinstance(x, str):
        return x
    elif isinstance(x, bytes):
        return str(x, 'utf-8')
    else:
        raise ValueError('Internal error')


@deprecated('Instead, use `delu.cuda.free_memory`.')
def free_memory(*args, **kwargs) -> None:
    """
    <DEPRECATION MESSAGE>
    """
    return free_memory_original(*args, **kwargs)


@deprecated('Instead, use functions from `torch.cuda`')
def get_gpus_info() -> Dict[str, Any]:
    """Get information about GPU devices: driver version, memory, utilization etc.

    <DEPRECATION MESSAGE>

    The example below shows what kind of information is returned as the result. All
    figures about memory are given in bytes.

    Returns:
        Information about GPU devices.

    Warning:
        The 'devices' value contains information about *all* gpus regardless of the
        value of ``CUDA_VISIBLE_DEVICES``.

    Examples:
        .. code-block::

            print(delu.hardware.get_gpu_info())

        Output example (formatted for convenience):

        .. code-block:: none

            {
                'driver': '440.33.01',
                'devices': [
                    {
                        'name': 'GeForce RTX 2080 Ti',
                        'memory_total': 11554717696,
                        'memory_free': 11554652160,
                        'memory_used': 65536,
                        'utilization': 0,
                    },
                    {
                        'name': 'GeForce RTX 2080 Ti',
                        'memory_total': 11552096256,
                        'memory_free': 11552030720,
                        'memory_used': 65536,
                        'utilization': 0,
                    },
                ],
            }
    """
    if pynvml is None:
        raise RuntimeError(
            'To use this function, install pynvml via `pip install pynvml<12.0`'
        )

    try:
        pynvml.nvmlInit()
    except NVMLError_LibraryNotFound as err:
        raise RuntimeError(
            'Failed to get information about GPU memory. '
            'Make sure that you actually have GPU and all relevant software installed.'
        ) from err
    n_devices = pynvml.nvmlDeviceGetCount()
    devices = []
    for device_id in range(n_devices):
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        devices.append(
            {
                'name': _to_str(pynvml.nvmlDeviceGetName(handle)),
                'memory_total': memory_info.total,
                'memory_free': memory_info.free,
                'memory_used': memory_info.used,
                'utilization': pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
            }
        )
    return {
        'driver': _to_str(pynvml.nvmlSystemGetDriverVersion()),
        'devices': devices,
    }
