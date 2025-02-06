



def draw_bounding_box(
        ax,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
        color: str = "cyan"
):
    '''
    Draw a bounding box in the matplotlib axis `ax`.
    '''

    ax.vlines(xmin, ymin, ymax, color = color)
    ax.vlines(xmax, ymin, ymax, color = color)
    ax.hlines(ymin, xmin, xmax, color = color)
    ax.hlines(ymax, xmin, xmax, color = color)