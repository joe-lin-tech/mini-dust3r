import rerun as rr
from argparse import ArgumentParser
import numpy as np
import open3d as o3d

parser = ArgumentParser("pointcloud vis script")
rr.script_add_args(parser)
args = parser.parse_args()
rr.script_setup(args, "mini-dust3r")

pcd = o3d.io.read_point_cloud("debug/pointcloud.ply")
positions = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

import ipdb; ipdb.set_trace()
rr.log(f"world/pointcloud", rr.Points3D(positions=positions, colors=colors))

rr.script_teardown(args)