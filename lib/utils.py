import numpy as np
import json

# read camera intrinsics
def get_camera_intrinsics(file_path):
    with open(file_path, 'r') as f:
        intrinsics = json.load(f)
    return intrinsics