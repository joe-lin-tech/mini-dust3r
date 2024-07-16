import point_cloud_utils as pcu
import numpy as np

v, _ = pcu.load_mesh_vf("debug/pointcloud.ply")
n_idx, n = pcu.estimate_point_cloud_normals_knn(v, 32)
print(n)