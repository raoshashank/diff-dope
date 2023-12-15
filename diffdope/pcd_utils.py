import torch
import meshcat
import meshcat.geometry as mcg
from typing import Tuple
import numpy as np
def meshcat_pcd_show(
        mc_vis: meshcat.Visualizer, point_cloud: np.ndarray, 
        color: Tuple[int]=None, name: str=None, 
        size: float=0.005, debug: bool=False) -> None:
    # color_orig = copy.deepcopy(color)
    if point_cloud.shape[0] != 3:
        point_cloud = point_cloud.swapaxes(0, 1)
    if color is None:
        color = np.zeros_like(point_cloud)
    else:
        # color = int('%02x%02x%02x' % color, 16)
        if not isinstance(color, np.ndarray):
            color = np.asarray(color).astype(np.float32)
        color = np.clip(color, 0, 255)
        color = np.tile(color.reshape(3, 1), (1, point_cloud.shape[1]))
        color = color.astype(np.float32)
    if name is None:
        name = 'scene/pcd'

    if debug:
        print("here in meshcat_pcd_show")
        from IPython import embed; embed()

    mc_vis[name].set_object(
        mcg.Points(
            mcg.PointsGeometry(point_cloud, color=(color / 255)),
            mcg.PointsMaterial(size=size)
    ))

def meshcat_frame_show(
        mc_vis: meshcat.Visualizer, name: str, 
        transform: np.ndarray=None, length: float=0.1, 
        radius: float=0.008, opacity: float=1.) -> None:
    """
    Initializes coordinate axes of a frame T. The x-axis is drawn red,
    y-axis green and z-axis blue. The axes point in +x, +y and +z directions,
    respectively.
    Args:
        mc_vis: a meshcat.Visualizer object.
        name: (string) the name of the triad in meshcat.
        transform (np.ndarray): 4 x 4 matrix representing the pose
        length: the length of each axis in meters.
        radius: the radius of each axis in meters.
        opacity: the opacity of the coordinate axes, between 0 and 1.
    """
    delta_xyz = np.array([[length / 2, 0, 0],
    [0, length / 2, 0],
    [0, 0, length / 2]])

    axes_name = ['x', 'y', 'z']
    colors = [0xff0000, 0x00ff00, 0x0000ff]
    rotation_axes = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]

    for i in range(3):
        material = meshcat.geometry.MeshLambertMaterial(
        color=colors[i], opacity=opacity)
        mc_vis[name][axes_name[i]].set_object(
        meshcat.geometry.Cylinder(length, radius), material)
        X = meshcat.transformations.rotation_matrix(
        np.pi/2, rotation_axes[i])
        X[0:3, 3] = delta_xyz[i]
        if transform is not None:
            X = np.matmul(transform, X)
        mc_vis[name][axes_name[i]].set_transform(X)

def transform_pcd_torch(pcd: torch.tensor, transform: torch.tensor) -> torch.tensor:
    assert (len(pcd.shape) == len(transform.shape)) and (len(pcd.shape)==3) 
    if pcd.shape[-1] != 4:
        pcd = torch.concatenate((pcd, torch.ones((pcd.shape[0],pcd.shape[1], 1)).to(pcd.device)), axis=-1)
    pcd_new = torch.bmm(transform,pcd.transpose(1,2))[:,:-1,:].transpose(1,2)
    return pcd_new

