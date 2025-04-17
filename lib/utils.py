import numpy as np
import json

def get_camera_intrinsics(file_path):
    """
    Load camera intrinsic parameters from a JSON file.

    The JSON is expected to contain keys: 'fx', 'fy', 'cx', 'cy'.

    Args:
        file_path (str): Path to the JSON file containing intrinsics.

    Returns:
        dict: Dictionary with intrinsic parameters.
              Example: {'fx': float, 'fy': float, 'cx': float, 'cy': float}
    """
    with open(file_path, 'r') as f:
        intrinsics = json.load(f)
    return intrinsics


def get_camera_extrinsics(file_path):
    """
    Load camera extrinsic matrix from a JSON file.

    The JSON is expected to contain a key 'cam2root' representing a 4x4 transformation matrix.

    Args:
        file_path (str): Path to the JSON file containing extrinsics.

    Returns:
        np.ndarray: 4x4 numpy array representing the transformation from camera to world coordinates.
    """
    with open(file_path, 'r') as f:
        extrinsics = json.load(f)
    return np.array(extrinsics["cam2root"])
