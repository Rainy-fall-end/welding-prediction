import numpy as np
import scipy as sp
from model.mesh import Mesh
# x, y, z are the 3D point coordinates
def b_spline_plane(mesh:Mesh,k=5):
    spline = sp.interpolate.Rbf(mesh.get_X(),mesh.get_Y(),mesh.deform,function='thin_plate',smooth=5, episilon=k)
    x_grid = np.linspace(0, 200, endpoint=True,num = 400)
    y_grid = np.linspace(0, 400, endpoint=True,num = 800)
    B1, B2 = np.meshgrid(x_grid, y_grid, indexing='xy')
    Z = spline(B1,B2)
    return spline,Mesh.fromfit(B1,B2,Z)