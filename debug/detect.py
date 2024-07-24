import numpy as np
import pandas as pd
from plyfile import PlyData
from mmdet3d.apis import init_model, inference_detector
import json

def convert_ply(input_path, output_path):
    plydata = PlyData.read(input_path)  # read file
    data = plydata.elements[0].data  # read data
    data_np = np.hstack([
        np.array(data['x'])[:, np.newaxis],
        np.array(data['y'])[:, np.newaxis],
        np.array(data['z'])[:, np.newaxis],
        np.ones_like(data['x'])[:, np.newaxis]
    ])
    data_np.astype(np.float32).tofile(output_path)

convert_ply('debug/pointcloud.ply', 'debug/pointcloud.bin')

config_file = 'debug/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py' # 'debug/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py' # 'debug/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py'
checkpoint_file = 'debug/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth' # 'debug/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth' # 'debug/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth'
model = init_model(config_file, checkpoint_file)
result = inference_detector(model, 'debug/pointcloud.bin')
result = result[0].pred_instances_3d

with open("debug/boxes.json", "w") as f:
    json.dump({
        "labels_3d": result['labels_3d'].cpu().numpy().tolist(),
        "scores_3d": result['scores_3d'].cpu().numpy().tolist(),
        "bboxes_3d": result['bboxes_3d'].tensor.cpu().numpy().tolist()
    }, f)
import ipdb; ipdb.set_trace()