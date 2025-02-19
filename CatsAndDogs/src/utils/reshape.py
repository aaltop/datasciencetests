'''
Functions to reshape torch tensors to a more flattened form.
'''

import torch

def items_per_pixel(t: torch.Tensor, channels_per_item) -> torch.Tensor:
    '''
    Transform a tensor (batches, N, width, height), such as one gotten
    from convolving a width-height image over N channels,
    to a form `[batches, width*height*(N//channels_per_item), channels_per_item]`.
    Effectively transforms the input such that each row contains one
    item.
    '''

    batches, N, width, height = map(int, t.shape)

    return (
        t
        .flatten(start_dim=2)
        .mT
        .reshape([batches,(N//channels_per_item)*width*height, channels_per_item])
    )