import torch

from contextlib import contextmanager

TORCH_DEVICE = "cpu"
if torch.cuda.is_available():
    TORCH_DEVICE = "cuda:0"

@contextmanager
def default_device(device: str):
    '''
    Context manager to set the global
    default device for torch.

    Yields function that can be used to move other tensors
    to the same device.
    '''

    original_device = torch.get_default_device()
    try:
        torch.set_default_device(device)
        yield lambda tens: tens.to(device)
    finally:
        torch.set_default_device(original_device)