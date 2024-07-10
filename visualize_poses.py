import numpy as np
import plotly.graph_objects as go

def plot_camera_poses(gt_poses, pred_poses):
    fig = go.Figure()

    # for pose in gt_poses:
    #     # Extract the translation part
    #     t = pose[:3, 3]

    #     # Extract the rotation part
    #     R = pose[:3, :3]
    #     x_axis = R[:, 0]
    #     y_axis = R[:, 1]
    #     z_axis = R[:, 2]

    #     # Scale the axis for better visualization
    #     scale = 1

    #     fig.add_trace(go.Scatter3d(x=[t[0]], y=[t[1]], z=[t[2]],
    #                                mode='markers',
    #                                marker=dict(size=5, color='red')))

    #     fig.add_trace(go.Scatter3d(x=[t[0], t[0] + scale * x_axis[0]],
    #                                y=[t[1], t[1] + scale * x_axis[1]],
    #                                z=[t[2], t[2] + scale * x_axis[2]],
    #                                mode='lines', line=dict(color='red')))

    #     fig.add_trace(go.Scatter3d(x=[t[0], t[0] + scale * y_axis[0]],
    #                                y=[t[1], t[1] + scale * y_axis[1]],
    #                                z=[t[2], t[2] + scale * y_axis[2]],
    #                                mode='lines', line=dict(color='green')))

    #     fig.add_trace(go.Scatter3d(x=[t[0], t[0] + scale * z_axis[0]],
    #                                y=[t[1], t[1] + scale * z_axis[1]],
    #                                z=[t[2], t[2] + scale * z_axis[2]],
    #                                mode='lines', line=dict(color='blue')))
        
    for pose in pred_poses:
        # Extract the translation part
        t = pose[:3, 3]

        # Extract the rotation part
        R = pose[:3, :3]
        x_axis = R[:, 0]
        y_axis = R[:, 1]
        z_axis = R[:, 2]

        # Scale the axis for better visualization
        scale = 0.1

        fig.add_trace(go.Scatter3d(x=[t[0]], y=[t[1]], z=[t[2]],
                                   mode='markers',
                                   marker=dict(size=5, color='blue')))

        fig.add_trace(go.Scatter3d(x=[t[0], t[0] + scale * x_axis[0]],
                                   y=[t[1], t[1] + scale * x_axis[1]],
                                   z=[t[2], t[2] + scale * x_axis[2]],
                                   mode='lines', line=dict(color='red')))

        fig.add_trace(go.Scatter3d(x=[t[0], t[0] + scale * y_axis[0]],
                                   y=[t[1], t[1] + scale * y_axis[1]],
                                   z=[t[2], t[2] + scale * y_axis[2]],
                                   mode='lines', line=dict(color='green')))

        fig.add_trace(go.Scatter3d(x=[t[0], t[0] + scale * z_axis[0]],
                                   y=[t[1], t[1] + scale * z_axis[1]],
                                   z=[t[2], t[2] + scale * z_axis[2]],
                                   mode='lines', line=dict(color='blue')))

    fig.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ))
    
    fig.show()

# Example usage
import pickle

with open("test.pkl", "rb") as f:
    gt_info = pickle.load(f)

with open("dust3r_output.pkl", "rb") as f:
    pred_info = pickle.load(f)

gt_poses = gt_info["camera"]["extrinsics"][::30]
pred_poses = pred_info["extrinsics"] 

plot_camera_poses(gt_poses, pred_poses)