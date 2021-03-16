import numpy as np


class OccupancyGrid(object):
    """OccupancyGrid class is a wrapper for the occupancy grid.
    """
    def __init__(self, output_shape):
        super().__init__()
        if isinstance(output_shape, str):
            output_shape = tuple(map(int, output_shape.split(",")))
        else:
            raise NotImplementedError()
        self._output_shape = output_shape

        # Array that contains the voxel centers

    @property
    def output_shape(self):
        return ((1,) + self._output_shape)

    def _bbox_from_points(self, pts):
        bbox = np.array([-0.51]*3 + [0.51]*3)
        step = (bbox[3:] - bbox[:3])/self._output_shape

        return bbox, step

    def voxelize(self, mesh):
        """Given a Mesh object we want to estimate the occupancy grid
        """
        pts = mesh.vertices.T
        # Make sure that the points have the correct shape
        assert pts.shape[0] == 3
        # Make sure that points lie in the unit cube
        if (np.any(np.abs(pts.min(axis=-1)) > 0.51) or
                np.any(pts.max(axis=-1) > 0.51)):
            raise Exception(
                "The points do not lie in the unit cube min:%r - max:%r"
                % (pts.min(axis=-1), pts.max(axis=-1))
            )

        bbox, step = self._bbox_from_points(pts)
        occupancy_grid = np.zeros(self._output_shape, dtype=np.float32)

        idxs = ((pts.T - bbox[:3].T)/step).astype(np.int32)
        # Array that will contain the occupancy grid, namely if a point lies
        # within a voxel then we assign one to this voxel, otherwise we assign
        # 0
        occupancy_grid[idxs[:, 0], idxs[:, 1], idxs[:, 2]] = 1.0
        return occupancy_grid[np.newaxis]
