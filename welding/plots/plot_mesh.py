import numpy as np
import matplotlib.pyplot as plt
from model.mesh import Mesh
from matplotlib import cm
class plot_func:
    @staticmethod  
    def plot_mesh(mesh:Mesh,save_path):
        plt.style.use('_mpl-gallery')
        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        triangulation = ax.plot_trisurf(mesh.xy_grid[:,0], mesh.xy_grid[:,1], mesh.deform, cmap='viridis',antialiased=True)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Value Label')
        fig.colorbar(triangulation, ax=ax, shrink=0.7, aspect=10)
        plt.savefig(save_path,dpi = 500, bbox_inches='tight')
    @staticmethod
    def plot_res_mesh(mesh1:Mesh,mesh2:Mesh,save_path):
        plt.style.use('_mpl-gallery')
        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        triangulation = ax.plot_trisurf(mesh1.xy_grid[:,0], mesh1.xy_grid[:,1], mesh1.deform-mesh2.deform, cmap='viridis',antialiased=True)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Value Label')
        fig.colorbar(triangulation, ax=ax, shrink=0.7, aspect=10)
        plt.savefig(save_path,dpi = 500, bbox_inches='tight')
    @staticmethod
    def plot_res_mesh_spline(mesh:Mesh,spline,save_path):
        B1,B2 = mesh.get_xy_grid2D()
        Z = spline(B1,B2)
        plt.style.use('_mpl-gallery')
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        triangulation = ax.plot_trisurf(mesh.xy_grid[:,0], mesh.xy_grid[:,1], mesh.deform-Z.ravel(), cmap='viridis',antialiased=True)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Value Label')
        fig.colorbar(triangulation, ax=ax, shrink=0.7, aspect=10)
        plt.savefig(save_path,dpi = 500, bbox_inches='tight')
        