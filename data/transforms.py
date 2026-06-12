"""
Module for assigning value to certain cells

Currently contains
- Centroid -> 1.0 cell + Surrounding Cells Reduced Punishment (value of surround_value)

Possible Future Works
- Bounding box size dependency
- Representing objects each as multiple 1.0 cells depending on bounding box dimensions 
"""

def paint_reduced_punishment(
    grid: np.ndarray,
    channel: int,
    center_y: int,
    center_x: int,
    radius: int = 1,
    surround_value: float = 0.3,
) -> None:
    '''
    Initial Condition:
    Before this function is used at all, grid is typically initially all 0.0.

    Input:
    grid - target tensor (all channels, modified in-place)
    channel - which channel to paint on (each object class gets its own)
    center x/y - centroid position on the grid
    radius - how many cells out from center to paint
    surround_value - what value to give cells surrounding centroid

    Effect:
    Paints a 1.0 in the centroid cell of the grid and paints the
    surrounding cells within `radius` with `surround_value`. Higher existing 
    values (like other overlapping centroids) are preserved.
    '''

    # Target center 
    grid[channel, center_y, center_x] = 1.0

    # Grid and Center Box Dimensions
    grid_h = grid.shape[1]
    grid_w = grid.shape[2]
    obj_hw_len = (2*radius) + 1

    # Top-left corner of the surrounding box
    pointer = [center_y - radius, center_x - radius]

    # Grab surrounding cells and ensure they are at least surround_value 
    # (clamp_ preserves higher values like existing 1.0s from overlapping centroids)
    grid[
        channel, 
        max(pointer[0], 0):min(pointer[0]+obj_hw_len, grid_h),   # start_y:end_y
        max(pointer[1], 0):min(pointer[1]+obj_hw_len, grid_w)    # start_x:end_x
    ].clamp_(min=surround_value)

