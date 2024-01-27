import numpy as np
class Mesh():
    def __init__(
        self,
        xy_grid: np.ndarray,
        deform: np.ndarray
    ):
        super(Mesh,self).__init__()
        self.xy_grid = xy_grid
        self.deform = deform
    @classmethod
    def fromfit(
        cls,
        B1:np.ndarray,
        B2:np.ndarray,
        Z:np.ndarray
    ):
        xy_grid = np.column_stack((B1.ravel(),B2.ravel()))
        deform = Z.ravel()
        return cls(xy_grid=xy_grid,deform=deform)
    def get_X(self):
        return self.xy_grid[:,0]
    def get_Y(self):
        return self.xy_grid[:,1]
    def get_xy_grid2D(self):
        x_size = len(sorted(set(self.get_X())))
        y_size = len(sorted(set(self.get_Y())))
        B1 = self.get_X().reshape(x_size,y_size)
        B2 = self.get_Y().reshape(x_size,y_size)
        return (B1.T,B2.T)