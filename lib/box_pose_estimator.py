import numpy as np
import open3d as o3d
import cv2

class BoxPoseEstimator:
    """
    Estimates 6D pose of boxes based on planar patches from point clouds.

    Attributes:
        fx (float): Focal length in x.
        fy (float): Focal length in y.
        cx (float): Optical center x.
        cy (float): Optical center y.
        extrinsics (np.ndarray): 4x4 transformation matrix from camera to world.
        known_dims (dict): Dictionary mapping class names to (x, y, z) box dimensions.
        poses_world (list): List of dictionaries with estimated poses.
    """

    def __init__(self, intrinsics, extrinsics, known_dims):
        """
        Initialize BoxPoseEstimator.

        Args:
            intrinsics (dict): Dictionary with keys 'fx', 'fy', 'cx', 'cy'.
            extrinsics (np.ndarray): 4x4 camera-to-world transformation.
            known_dims (dict): Dictionary of box class -> [x, y, z] dimensions in meters.
        """
        self.fx = intrinsics['fx']
        self.fy = intrinsics['fy']
        self.cx = intrinsics['cx']
        self.cy = intrinsics['cy']
        self.extrinsics = extrinsics
        self.known_dims = known_dims
        self.poses_world = []

    def estimate_pose_from_patch(self, patch, box_class):
        """
        Estimate 6D pose from a planar patch assumed to belong to a box surface.

        Args:
            patch (open3d.geometry.OrientedBoundingBox): The patch bounding box.
            box_class (str): The class name of the box.

        Returns:
            tuple: (center_position_world, rotation_matrix_world)
        """
        center_cam = patch.get_center()
        R_cam = patch.R

        known_dim = self.known_dims.get(box_class, [0.1, 0.1, 0.1])
        face_dims = {
            "top": sorted([known_dim[0], known_dim[2]]),
            "side": sorted([known_dim[1], known_dim[2]]),
            "front": sorted([known_dim[0], known_dim[1]])
        }

        patch_extent = sorted(patch.extent[:2])
        best_face = None
        min_diff = float('inf')

        for face, dims in face_dims.items():
            diff = np.linalg.norm(np.array(patch_extent) - np.array(dims))
            if diff < min_diff:
                min_diff = diff
                best_face = face

        # Compute rotation alignment based on face type
        if best_face == "top":
            R_base = R_cam
            expected_dims = sorted([known_dim[0], known_dim[2]])
            patch_dims = sorted(patch.extent[:2])
            flip = np.linalg.norm(np.array(patch_dims) - np.array(expected_dims[::-1])) < \
                   np.linalg.norm(np.array(patch_dims) - np.array(expected_dims))
            R_aligned = R_base @ (
                o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, np.pi/2]) if flip else np.eye(3)
            )
        elif best_face == "side":
            R_base = R_cam @ o3d.geometry.get_rotation_matrix_from_axis_angle([np.pi/2, 0, 0])
            expected_dims = sorted([known_dim[1], known_dim[2]])
            patch_dims = sorted(patch.extent[:2])
            flip = np.linalg.norm(np.array(patch_dims) - np.array(expected_dims[::-1])) < \
                   np.linalg.norm(np.array(patch_dims) - np.array(expected_dims))
            R_aligned = R_base @ (
                o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, np.pi/2]) if flip else np.eye(3)
            )
        elif best_face == "front":
            R_base = R_cam @ o3d.geometry.get_rotation_matrix_from_axis_angle([0, -np.pi/2, 0])
            expected_dims = sorted([known_dim[0], known_dim[1]])
            patch_dims = sorted(patch.extent[:2])
            flip = np.linalg.norm(np.array(patch_dims) - np.array(expected_dims[::-1])) < \
                   np.linalg.norm(np.array(patch_dims) - np.array(expected_dims))
            R_aligned = R_base @ (
                o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, np.pi/2]) if flip else np.eye(3)
            )
        else:
            R_aligned = R_cam

        center_world = (self.extrinsics @ np.append(center_cam, 1))[:3]
        R_world = self.extrinsics[:3, :3] @ R_aligned

        self.poses_world.append({
            "class": box_class,
            "position": center_world,
            "rotation_matrix": R_world,
            "dimensions": known_dim,
            "face": best_face
        })

        return center_world, R_world

    def estimate_from_all(self, patch_list):
        """
        Estimate poses for all patches in the list.

        Args:
            patch_list (list): List of dictionaries with 'points' and 'class' keys.

        Returns:
            list: Estimated poses with world coordinates and rotation matrices.
        """
        self.poses_world.clear()
        for patch_dict in patch_list:
            patch = patch_dict['points']
            cls = patch_dict['class']
            self.estimate_pose_from_patch(patch, cls)
        return self.poses_world

    def print_poses(self):
        """
        Print all estimated poses in a readable format to the console.
        """
        for i, pose in enumerate(self.poses_world):
            print(f"Box {i} ({pose['class']}) - face: {pose['face']}")
            print(f"  Position: {pose['position']}")
            print(f"  Rotation matrix:\n{pose['rotation_matrix']}")
            print(f"  Dimensions: {pose['dimensions']}\n")

    def visualize_poses(self, image=None, pointcloud=None, colorize_pc_with_image=False):
        """
        Visualize estimated poses with Open3D coordinate frames and spheres.

        Args:
            image (np.ndarray, optional): Not currently used.
            pointcloud (open3d.geometry.PointCloud, optional): Optional background point cloud.
            colorize_pc_with_image (bool): Unused option for future extension.
        """
        geometries = []

        if pointcloud is not None:
            geometries.append(pointcloud)

        for pose in self.poses_world:
            center = pose['position']
            R = pose['rotation_matrix']
            dims = pose['dimensions']
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = center

            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
            frame.transform(T)
            geometries.append(frame)

            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            sphere.translate(center)
            sphere.paint_uniform_color([0.8, 0.1, 0.1])
            geometries.append(sphere)

        o3d.visualization.draw_geometries(geometries, window_name="Estimated Box Poses")
