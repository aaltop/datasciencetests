import torch

def bounding_box_corners(bb: torch.Tensor) -> torch.Tensor:
    '''
    Given the bounding box `bb` (xmin, ymin, xmax, ymax), return
    the corners, starting from top left in clockwise manner.

    Multiple boxes can be passed, with one box per row. For a `bb`
    of size (N, 4), the result is (N, 4, 2), where the last dimension
    has (x,y).
    '''

    # account for bb potentially being row vector
    bb = bb.reshape([-1,4])

    xmin, ymin, xmax, ymax = bb.T

    corners = torch.vstack([
        xmin, ymin,
        xmax, ymin,
        xmax, ymax,
        xmin, ymax
    ]).T
    corners = corners.reshape([-1, 4, 2])

    return corners

def bounding_box_area(bb: torch.Tensor) -> torch.Tensor:
    '''
    Calculate bounding box area. For `bb` of shape
    (N,4) with rows (xmin, ymin, xmax, ymax), the result
    is of shape (N).
    '''

    # account for bb potentially being row vector
    bb = bb.reshape([-1, 4])

    widths = bb[:,2] - bb[:,0]
    heights = bb[:,-1] - bb[:,1]

    return widths*heights

# Checking for corner's inclusion:
# x coordinate >= xmin && <= xmax 
# AND y coordinate >= ymin && <= ymax.
#
# Calculate (?): all(([xmin, xmax] - x)*[-1,1] > 0)
#
# vectorise calculation (for x):
# bounds = torch.hstack([xmins, xmaxs])
# exes = x.repeat([bounds.shape[0], 2])
# mult = torch.tensor([-1,1]).repeat([bounds.shape[0], 1])
# between = torch.sign((bounds - exes)*mult)
# # the sum here should be either zero or two (per row)
# is_between = (between[:,0] + between[:,1]) > 0
#
# here, the whole calculation can be reused for one other corner,
# bounds can be reused for the other x value, mult is reusable
# for all calculations.
def included_corners(bb1: torch.Tensor, bb2: torch.Tensor) -> torch.Tensor:
    '''
    For bounding boxes `bb1`, (xmin, ymin, xmax, ymax), (N, 4), and
    `bb2`, (M, 4), return a boolean matrix (4, M, N), where index (i, j, k) denotes
    whether the corner i (from top left clockwise)
    of box j from `bb2` is within box k from `bb1` (including on the border).
    '''

    xmin, ymin, xmax, ymax = bb1.T

    N = len(bb1)
    M = len(bb2)
    x_bounds = torch.stack([xmin, xmax]).T.repeat(M, 1, 1)
    y_bounds = torch.stack([ymin, ymax]).T.repeat(M, 1, 1)
    mult = torch.tensor([-1,1]).repeat(M, N, 1)

    xmin, ymin, xmax, ymax = bb2.T

    bools = []
    for vals, bounds in (xmin,x_bounds), (xmax, x_bounds), (ymin, y_bounds), (ymax, y_bounds):

        between = torch.sign((
            bounds 
            - vals.repeat_interleave(N*2).reshape([M, N, 2])
        )*mult)

        bools.append(
            (between[:,:,0] + between[:,:,1]) > 0
        )

    tand = torch.logical_and
    corner_bools = [
        tand(bools[0], bools[2]),
        tand(bools[1], bools[2]),
        tand(bools[1], bools[-1]),
        tand(bools[0], bools[-1])
    ]
    return torch.stack(corner_bools)  


# Possible cases of intersection:
# - exact overlap, intersection is area of either one
# - one inside the other, intersection is area of one inside
# - no overlap, intersection is zero
# - partial overlap, more complex
#
# From a different perspective: the corners of one box can either be:
# - all inside the other (if >=3 corners are inside)
# - two inside the other (area is height/width difference of the two
# inside corners times width/height difference of one of the inside corners and
# the opposite corner of the other box)
# - one inside the other (e.g. topleft inside -> area from inside topleft
# to outside bottomright)
# - none inside, in which case the other box is inside this one 
# (just test for all inside as above), the other box has one side inside
# the other, or the two do not overlap.
#
# keep in mind the possibility of a corner being exactly on the border.
def intersection(bb1: torch.Tensor, bb2: torch.Tensor) -> torch.Tensor:
    '''
    Calculate the intersection of the bounding boxes `bb1` and `bb2` of
    shape (N, 4) and (M, 4), respectively, with a bounding box (xmin, ymin, xmax, ymax)
    on each row. Return the intersection areas as (M, N).

    NOTE: This is based on inclusion of `bb2`'s corners inside the boxes
    of `bb1`: only if a box in `bb2` has corners inside a box in `bb1`
    will the intersection be calculated. For example, if a box in `bb1`
    is completely inside a box in `bb2`, the intersection area returned will
    be 0.0, though the actual intersection area is the area of the `bb1`
    box, because none of the `bb2` box's corners are within the `bb1` box. 
    To get all areas, run the function switching the place
    of `bb1` and `bb2` and combine the results, or preferably use
    `symmetric_intersection()`.
    '''

    corners = included_corners(bb1, bb2)
    # how many of the corners are within the other box
    corners_in = corners.sum(dim=0)

    all_corner = corners_in > 2
    two_corner = corners_in == 2
    one_corner = corners_in == 1
    areas = torch.zeros([len(bb1), len(bb2)], dtype=bb1.dtype)
    # all corners
    # -----------
    areas[all_corner.T] = bounding_box_area(bb2).repeat((len(bb1), 1))[all_corner.T]

    # two corners
    # -----------

    def logical_and(*tensors):

        result = tensors[0]
        for tens in tensors[1:]:
            result = torch.logical_and(result, tens)

        return result
    
    bb2_xmin, bb2_ymin, bb2_xmax, bb2_ymax = (
        bb2[:,0],
        bb2[:,1],
        bb2[:,2],
        bb2[:,3]
    )
    bb2_widths = bb2_xmax - bb2_xmin
    bb2_heights = bb2_ymax - bb2_ymin
        
    if torch.any(two_corner):
        # the top side is within the box
        top = logical_and(corners[0,:,:], corners[1,:,:], two_corner).T
        if torch.any(top):
            # width times height
            areas[top] = (bb2_widths*(bb1[:,-1].reshape((-1,1)) - bb2_ymin))[top]

        right = logical_and(corners[1,:,:], corners[2,:,:], two_corner).T
        if torch.any(right):
            areas[right] = ((bb2_xmax - bb1[:,0].reshape((-1,1)))*bb2_heights)[right]

        bottom = logical_and(corners[2,:,:], corners[3,:,:], two_corner).T
        if torch.any(bottom):
            areas[bottom] = (bb2_widths*(bb2_ymax - bb1[:, 1].reshape((-1,1))))[bottom]

        left = logical_and(corners[3,:,:], corners[0,:,:], two_corner).T
        if torch.any(left):
            areas[left] = ((bb1[:,2].reshape((-1,1)) - bb2_xmin)*bb2_heights)[left]

    # one corner
    # ----------

    if torch.any(one_corner):
        # top left corner is inside the box
        tl = logical_and(corners[0,:,:], one_corner).T
        if torch.any(tl):
            # width times height
            areas[tl] = (
                (bb1[:,2].reshape((-1,1)) - bb2_xmin)
                *(bb1[:,-1].reshape((-1,1)) - bb2_ymin)
            )[tl]
        
        tr = logical_and(corners[1,:,:], one_corner).T
        if torch.any(tr):
            areas[tr] = (
                (bb2_xmax - bb1[:,0].reshape((-1,1)))
                *(bb1[:,-1].reshape((-1,1)) - bb2_ymin)
            )[tr]

        br = logical_and(corners[2,:,:], one_corner).T
        if torch.any(br):
            areas[br] = (
                (bb2_xmax - bb1[:,0].reshape((-1,1)))
                *(bb2_ymax - bb1[:,1].reshape((-1,1)))
            )[br]

        bl = logical_and(corners[3,:,:], one_corner).T
        if torch.any(bl):
            areas[bl] = (
                (bb1[:,2].reshape((-1,1)) - bb2_xmin)
                *(bb2_ymax - bb1[:,1].reshape((-1,1)))
            )[bl]
    

    return areas.T

def symmetric_intersection(bb1: torch.Tensor, bb2: torch.Tensor) -> torch.Tensor:
    '''
    Calculate the intersection of the bounding boxes `bb1` and `bb2` of
    shape (N, 4) and (M, 4), respectively, with a bounding box (xmin, ymin, xmax, ymax)
    on each row. Return the intersection areas as (M, N).

    Unlike `intersection()`, the result of this calculation is symmetric
    in the arguments -- aside from the result being transposed depending
    on the order of the arguments -- and therefore gives all the intersection
    areas.
    '''

    one_to_two = intersection(bb1, bb2)
    if torch.any(one_to_two == 0.0):
        two_to_one = intersection(bb2, bb1)
        return torch.where(one_to_two != 0.0, one_to_two, two_to_one.T)
    else:
        return one_to_two



def intersection_over_union(bb1: torch.Tensor, bb2: torch.Tensor, dtype = torch.float32) -> torch.Tensor:
    '''
    Calculate the intersection-over-union AKA Jaccard index of
    the bounding boxes `bb1` (N,4), and `bb2`, (M,4). The format of a bounding
    box should be (xmin, ymin, xmax, ymax).

    Return the values as (M,N).
    '''

    intersections = symmetric_intersection(bb1.to(dtype), bb2.to(dtype))

    bb1_areas = bounding_box_area(bb1)
    bb2_areas = bounding_box_area(bb2).reshape([-1,1])
    areas_sum = bb2_areas + bb1_areas

    return intersections/(areas_sum - intersections)
    