import numpy as np
import pandas as pd
from plyfile import PlyData
from mmdet3d.apis import init_model, inference_detector

def convert_ply(input_path, output_path):
    plydata = PlyData.read(input_path)  # read file
    data = plydata.elements[0].data  # read data
    data_pd = pd.DataFrame(data)  # convert to DataFrame
    data_np = np.zeros(data_pd.shape, dtype=np.float)  # initialize array to store data
    property_names = data[0].dtype.names  # read names of properties
    for i, name in enumerate(
            property_names):  # read data by property
        data_np[:, i] = data_pd[name]
    data_np.astype(np.float32).tofile(output_path)

convert_ply('debug/pointcloud.ply', 'debug/pointcloud.bin')

config_file = 'debug/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py' # 'debug/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py' # 'debug/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py'
checkpoint_file = 'debug/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth' # 'debug/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth' # 'debug/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth'
model = init_model(config_file, checkpoint_file)
result = inference_detector(model, 'debug/pointcloud.bin')
import ipdb; ipdb.set_trace()