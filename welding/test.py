from model.utils import load_data
from model.mesh import Mesh
from plots.plot_mesh import plot_func
from B_spline.plane import b_spline_plane
import numpy as np
welding_path = "D:\\User\\Jinke\\welding\\data\\welding.npy"
para_path = "D:\\User\\Jinke\\welding\\data\\para.npy"
x_data,y_data,grid = load_data(welding_path,para_path)
mesh_test = Mesh(xy_grid=grid,deform=y_data[8])
mesh_fit_spline,mesh_fit = b_spline_plane(mesh_test)
plot_func.plot_mesh(mesh_test,"D:\\User\\Jinke\\welding\\images\\mesh_1.jpg")
plot_func.plot_res_mesh_spline(mesh_test,mesh_fit_spline,"D:\\User\\Jinke\\welding\\images\\mesh_res1.jpg")