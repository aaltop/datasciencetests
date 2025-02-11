'''
Functions to reshape torch tensors to a more flattened form.
'''

import torch

def items_per_pixel(t: torch.Tensor, channels_per_item) -> torch.Tensor:
    '''
    Transform a tensor (N, width, height), such as one gotten
    from convolving a width-height image over N channels,
    to a form `[width*height*(N//channels_per_item), channels_per_item]`.
    Effectively transforms the input such that each row contains one
    item.
    '''

    N, width, height = map(int, t.shape)

    return (
        t
        .flatten(start_dim=1)
        .T
        .reshape([(N//channels_per_item)*width*height, channels_per_item])
    )