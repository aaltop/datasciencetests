'''
Functions to calculate the "default boxes" as described in
**SSD: Single Shot MultiBox Detector** by Liu *et al.*. 
'''

import torch

def scales(num_layers, s_min = 0.2, s_max = 0.9) -> list[float]:
    '''
    Calculate the scale for each layer out of `num_layers` linearly
    from `s_min` > 0 to `s_max` <= 1.

    The output has the scale for layer `i` at index `i` in [0, num_layers).
    '''

    m = num_layers

    return [
        s_min + (s_max - s_min)/(m - 1)*(k - 1)
        for k in range(1, m + 1)
    ]

def default_box_centers(width, height) -> torch.Tensor:
    '''
    Calculate the centers of each pixel in a `width`-by-`height`
    image, used as the centers of the default boxes.

    Return center values normalised by the width and height (i.e. restricted to
    [0,1]). For return `ret`, `ret[i]` is the center of the i-th pixel
    when iterating over all pixels row-wise (left-to-right, top-to-bottom).
    '''

    x_pixels = (torch.arange(start = 0, end = width)+0.5)/width
    y_pixels = (torch.arange(start = 0, end = height)+0.5)/height

    x, y = torch.meshgrid(x_pixels, y_pixels, indexing = "xy")
    return torch.vstack([x.flatten(), y.flatten()]).T

def default_boxes(scale, centers, ratios:list[float] | None = None) -> torch.Tensor:
    '''
    Calculate the default boxes as (x_min, y_min, x_max, y_max).
    
    Return tensor with first index iterating over pixels,
    second index iterating over the box values, and third over
    the ratios in order.

    `scale`: Scales of the boxes, (0,1].

    `centers`: as per `default_box_centers()`.

    `ratios`: ratios of the boxes' widths and heights.
    '''

    if ratios is None:
        ratios = [1,2,3,1/2,1/3]

    ratios_sqrt = torch.sqrt(torch.tensor(ratios))

    box_half_w = scale*ratios_sqrt*0.5
    box_half_h = scale/ratios_sqrt*0.5
    # offsets for each ratio, the first four will be for the first
    # ratio, 5-8 for the second (if 2+ ratios), and so on
    offsets = torch.stack([
        -box_half_w,
        -box_half_h,
        box_half_w,
        box_half_h
    ], dim=1).flatten()

    # x,y,x,y
    x_and_y = centers.repeat(1,2)
    # consider x_and_y as a "single" value, then repeat that value
    # on one row (as in, don't repeat), and len(ratios) times on
    # the columns; if x_and_y.shape = (row,col), this gives a shape
    # of (row, len(ratios)*col).
    x_and_y = x_and_y.repeat(1,len(ratios))

    # rows have pixels, each group of four columns has one box definition
    boxes = x_and_y + offsets.repeat(len(x_and_y),1)
    # first index pixels, second boxes for pixel, third specific ratio box
    boxes = boxes.reshape([-1, len(ratios), 4])

    return boxes

