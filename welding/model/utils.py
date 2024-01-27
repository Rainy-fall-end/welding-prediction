import numpy as np
from scipy.spatial import KDTree
def load_data(welding_path,para_path,x_size=200,y_size=400,design = 6):
    welding = np.load(welding_path)
    para = np.load(para_path)
    ## xdata (n,[params])
    x_data = para
    ## ydata (n,grid,deform)
    
    x_grid = np.arange(0, x_size+1, 10)
    y_grid = np.arange(0, y_size+1, 10)
    # x_grid_1 = np.arange(0, 20, 2)
    # x_grid_2 = np.arange(20, x_size+1, 10)
    # x_grid = np.concatenate((x_grid_1,x_grid_2))
    # y_grid = np.arange(0, y_size+1, 10)
    xy_grid = np.array(np.meshgrid(x_grid, y_grid)).T.reshape(-1, 2)
    y_data = np.zeros((welding.shape[0],len(xy_grid)))
    for i in range(welding.shape[0]):
        welding[i,:,0] = welding[i,:,0] - para[i,1]*0.5
        kdTree = KDTree(welding[i,:,0:2])
        near_points = kdTree.query(xy_grid)[1]
        y_data[i] = welding[i,:,2][near_points]+welding[i,:,3][near_points] - design
    return (x_data,y_data,xy_grid)


