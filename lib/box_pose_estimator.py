import numpy as np
import open3d as o3d
import cv2

class BoxPoseEstimator:
    def __init__(self, intrinsics, extrinsics, known_dims):
        self.fx = intrinsics['fx']
        self.fy = intrinsics['fy']
        self.cx = intrinsics['cx']
        self.cy = intrinsics['cy']
        self.extrinsics = extrinsics  # 4x4 camera-to-world
        self.known_dims = known_dims
        self.poses_world = []

    def estimate_pose_from_patch(self, patch, box_class):
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

        if best_face == "top":
            # Take R_base as the rotation with Z pointing up
            R_base = R_cam
            # sort the 2 other dimensions (length and depth)
            expected_dims = sorted([known_dim[0], known_dim[2]])
            # sort the 2 dimensions of the patch
            patch_dims = sorted(patch.extent[:2])
            flip = np.linalg.norm(np.array(patch_dims) - np.array(expected_dims[::-1])) < np.linalg.norm(np.array(patch_dims) - np.array(expected_dims))
            R_aligned = R_base @ (o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, np.pi/2]) if flip else np.eye(3))
        elif best_face == "side":
            R_base = R_cam @ o3d.geometry.get_rotation_matrix_from_axis_angle([np.pi/2, 0, 0])
            expected_dims = sorted([known_dim[1], known_dim[2]])
            patch_dims = sorted(patch.extent[:2])
            flip = np.linalg.norm(np.array(patch_dims) - np.array(expected_dims[::-1])) < np.linalg.norm(np.array(patch_dims) - np.array(expected_dims))
            R_aligned = R_base @ (o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, np.pi/2]) if flip else np.eye(3))
        elif best_face == "front":
            R_base = R_cam @ o3d.geometry.get_rotation_matrix_from_axis_angle([0, -np.pi/2, 0])
            expected_dims = sorted([known_dim[0], known_dim[1]])
            patch_dims = sorted(patch.extent[:2])
            flip = np.linalg.norm(np.array(patch_dims) - np.array(expected_dims[::-1])) < np.linalg.norm(np.array(patch_dims) - np.array(expected_dims))
            R_aligned = R_base @ (o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, np.pi/2]) if flip else np.eye(3))
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
        self.poses_world.clear()
        for patch_dict in patch_list:
            patch = patch_dict['points']
            cls = patch_dict['class']
            self.estimate_pose_from_patch(patch, cls)
        return self.poses_world

    def print_poses(self):
        for i, pose in enumerate(self.poses_world):
            print(f"Box {i} ({pose['class']}) - face: {pose['face']}")
            print(f"  Position: {pose['position']}")
            print(f"  Rotation matrix:\n{pose['rotation_matrix']}")
            print(f"  Dimensions: {pose['dimensions']}\n")

    def visualize_poses(self, image=None, pointcloud=None, colorize_pc_with_image=False):
        geometries = []
        image_proj = image.copy() if image is not None else None()
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

            if image is not None:
                # Create box from pose to get box_corners
                box = o3d.geometry.OrientedBoundingBox(center, R, dims)
                box_corners = np.asarray(box.get_box_points())
                for corner in box_corners:
                    x, y, z = corner
                    if z > 0:
                        u = int((x * self.fx) / z + self.cx)
                        v = int((y * self.fy) / z + self.cy)
                        if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
                            cv2.circle(image_proj, (u, v), 3, (0, 255, 0), -1)
                lines = np.asarray(o3d.geometry.LineSet.create_from_oriented_bounding_box(box).lines)
                for i, j in lines:
                    p1 = box_corners[i]
                    p2 = box_corners[j]
                    if p1[2] > 0 and p2[2] > 0:
                        u1, v1 = int((p1[0]*self.fx)/p1[2]+self.cx), int((p1[1]*self.fy)/p1[2]+self.cy)
                        u2, v2 = int((p2[0]*self.fx)/p2[2]+self.cx), int((p2[1]*self.fy)/p2[2]+self.cy)
                        if all(0 <= val < image.shape[1] for val in [u1, u2]) and all(0 <= val < image.shape[0] for val in [v1, v2]):
                            cv2.line(image_proj, (u1, v1), (u2, v2), (255, 0, 0), 1)

        if pointcloud is not None:
            if colorize_pc_with_image and image is not None:
                inv_ext = np.linalg.inv(self.extrinsics)
                points_world = np.asarray(pointcloud.points)
                ones = np.ones((points_world.shape[0], 1))
                points_h = np.hstack([points_world, ones])
                points_cam = (inv_ext @ points_h.T).T[:, :3]
                colors = []
                for pt in points_cam:
                    if pt[2] <= 0:
                        colors.append([0, 0, 0])
                        continue
                    u = int((pt[0] * self.fx) / pt[2] + self.cx)
                    v = int((pt[1] * self.fy) / pt[2] + self.cy)
                    if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
                        color = image[v, u, :] / 255.0
                    else:
                        color = [0, 0, 0]
                    colors.append(color)
                pointcloud.colors = o3d.utility.Vector3dVector(np.array(colors))
            geometries.insert(0, pointcloud)

        o3d.visualization.draw_geometries(geometries, window_name="6D Box Poses")

        if image is not None:
            cv2.imshow("Projected 3D Box", image_proj)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
