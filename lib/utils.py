import numpy as np
import json

# read camera intrinsics
def get_camera_intrinsics(file_path):
    with open(file_path, 'r') as f:
        intrinsics = json.load(f)
    return intrinsics


def get_camera_extrinsics(file_path):
    with open(file_path, 'r') as f:
        extrinsics = json.load(f)
    return np.array(extrinsics["cam2root"])