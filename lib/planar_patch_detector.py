import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import cv2

class PlanarPatchDetector:
    def __init__(self, color_img, depth_img, intrinsics, known_dims, depth_scale=1000.0, expand_ratio=0.1):
        self.color_img = color_img
        self.depth_img = depth_img
        self.fx = intrinsics['fx']
        self.fy = intrinsics['fy']
        self.cx = intrinsics['cx']
        self.cy = intrinsics['cy']
        self.depth_scale = depth_scale
        self.expand_ratio = expand_ratio
        self.known_dims = known_dims

        self.detections = []
        self.per_box_pcds = []
        self.per_box_patches = []
        self.cleaned_box_surfaces = []

        # Full scene point cloud without correction
        self.full_scene_pcd = self.create_full_pointcloud()

    def voxel_downsampling(self, pcd, voxel_size):
        """Downsample the point cloud using a voxel grid filter."""
        pcd_downsampled = pcd.voxel_down_sample(voxel_size)
        return pcd_downsampled

    def create_full_pointcloud(self):
        """Converts every pixel in the depth image to a 3D point using a pinhole camera model"""
        z = self.depth_img.astype(float) / self.depth_scale
        h, w = self.depth_img.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd.remove_non_finite_points()

    def pointcloud_to_surface_mesh(self, pcd):
        """
        Convert a dense point cloud into a surface mesh using Poisson reconstruction.
        Works best if the point cloud represents a flat or smooth surface.
        """
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
        return mesh

    def set_yolo_detections(self, detections):
        """Detections format: list of (class, conf, (x1, y1, x2, y2))"""
        self.detections = detections

    def expand_bbox(self, x, y, w, h):
        """Extend the area of bbox"""
        dh, dw = int(h * self.expand_ratio), int(w * self.expand_ratio)
        img_h, img_w = self.depth_img.shape
        x_new = max(x - dw, 0)
        y_new = max(y - dh, 0)
        x2_new = min(x + w + dw, img_w)
        y2_new = min(y + h + dh, img_h)
        return x_new, y_new, x2_new - x_new, y2_new - y_new

    def crop_pointcloud_from_bbox(self, x, y, w, h):
        """Converts the cropped image region to a point cloud"""
        depth_crop = self.depth_img[y:y+h, x:x+w].astype(float) / self.depth_scale
        rows, cols = np.indices((h, w))
        z = depth_crop
        x3d = (cols + x - self.cx) * z / self.fx
        y3d = (rows + y - self.cy) * z / self.fy
        points = np.stack((x3d, y3d, z), axis=-1).reshape(-1, 3)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd.remove_non_finite_points()

    def compute_normals(self, pcd, voxel_size=0.005):
        """Compute normals from the points"""
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
        if len(pcd.points) > 300:
            pcd, _ = pcd.remove_radius_outlier(nb_points=16, radius=0.02)
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=50))
        pcd.orient_normals_to_align_with_direction([0, 0, -1])
        return pcd

    def detect_planar_patches(self, pcd, normal_variance_threshold_deg=60, coplanarity_deg=60,
                              min_plane_edge_length=0.03, min_num_points=20, knn=50):
        return pcd.detect_planar_patches(
            normal_variance_threshold_deg=normal_variance_threshold_deg,
            coplanarity_deg=coplanarity_deg,
            min_plane_edge_length=min_plane_edge_length,
            min_num_points=min_num_points,
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn)
        )


    def fit_plane_to_points(self, pcd_points, distance_threshold=0.01):
        """Fit a plane to the given points using RANSAC"""
        plane_model, inliers = pcd_points.segment_plane(distance_threshold=distance_threshold, ransac_n=3, num_iterations=1000)
        [a, b, c, d] = plane_model

        # Extract the inlier points
        inlier_points = pcd_points.select_by_index(inliers)
        return inlier_points, plane_model


    def process_all(self, visualize=False, visualize_individual=False, mode='3d',
                    normal_variance_threshold_deg=60, coplanarity_deg=60,
                    min_plane_edge_length=0.03, min_num_points=20, knn=50,
                    voxel_size=0.01, distance_threshold=0.01, nb_neighbors=50, std_ratio=2.0,
                    min_plane_area=0.1):  # Add the min_plane_area parameter
        """Run the full pipeline per detection and visualize on the full corrected point cloud."""
        self.per_box_pcds = []
        self.per_box_patches = []
        all_patches = []

        for i, (cls, conf, (x1, y1, x2, y2)) in enumerate(self.detections):
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            w, h = x2 - x1, y2 - y1
            x_exp, y_exp, w_exp, h_exp = self.expand_bbox(x1, y1, w, h)
            x_exp, y_exp, w_exp, h_exp = map(int, (x_exp, y_exp, w_exp, h_exp))

            pcd = self.crop_pointcloud_from_bbox(x_exp, y_exp, w_exp, h_exp)
            pcd = self.compute_normals(pcd)

            # Apply voxel downsampling to reduce points and noise
            pcd = self.voxel_downsampling(pcd, voxel_size=voxel_size)

            # Apply statistical outlier removal for additional noise reduction
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

            # Fit a plane to the points and extract inliers
            inlier_points, plane_model = self.fit_plane_to_points(pcd, distance_threshold=distance_threshold)

            # Optional stricter filtering after RANSAC
            filtered_points = self.filter_points_on_plane(inlier_points, plane_model, threshold=distance_threshold)

            # Store the cleaned point cloud for this detection
            self.cleaned_box_surfaces.append(filtered_points)


            patches = self.detect_planar_patches(
                pcd,
                normal_variance_threshold_deg=normal_variance_threshold_deg,
                coplanarity_deg=coplanarity_deg,
                min_plane_edge_length=min_plane_edge_length,
                min_num_points=min_num_points,
                knn=knn
            )

            self.per_box_pcds.append(pcd)
            self.per_box_patches.append(patches)

            print(f"\n Detection {i} (class: {cls}) - found {len(patches)} planar patches")

            colors = plt.cm.get_cmap("Set1", 10).colors  # Use high-contrast color map

            patch_normals = [patch.R[:, 2] for patch in patches]
            dot_products = [abs(np.dot(normal, [0, 0, 1])) for normal in patch_normals]

            expected_dims = self.known_dims.get(cls, [0.3, 0.2, 0.1])
            expected_area = expected_dims[0] * expected_dims[1]
            patch_scores = []
            for j, dp in enumerate(dot_products):  # enhanced scoring with size prior
                patch = patches[j]
                normal_alignment_score = 1 - dp  # how close the normal is to being vertical
                inlier_count = len(patch.get_point_indices_within_bounding_box(patch.get_box_points()))
                patch_extent = patch.extent
                patch_area = np.prod(sorted(patch_extent)[-2:])
                area_score = 1 - abs(patch_area - expected_area) / expected_area
                patch_scores.append((normal_alignment_score + area_score, inlier_count))

            for idx, (normal, (score, inliers)) in enumerate(zip(patch_normals, patch_scores)):
                print(f"   - Patch {idx}: normal={normal}, alignment_score={score:.3f}, inliers={inliers}")

            if patches:
                # Apply area filtering: remove patches smaller than the minimum area threshold
                patches = [patch for patch in patches if np.prod(patch.extent) >= min_plane_area]

                # Ensure we have patches left after filtering
                if len(patches) > 0:
                    # Keep only the best patches based on size and flatness
                    top_patch_indices = sorted(range(len(patches)), key=lambda j: (patch_scores[j][0], patch_scores[j][1]), reverse=True)

                    # Ensure top_patch_indices doesn't exceed the number of available patches
                    top_patch_indices = top_patch_indices[:min(3, len(patches))]  # Limit to 3 patches or less

                    # Process the selected patches
                    for j in top_patch_indices:
                            patch = patches[j]
                            patch.color = np.array(colors[j % len(colors)][:3])  # Assign color to patch
                            all_patches.append(patch)
                    else:
                        print(f"No valid patches left after filtering (small planes).")

                # Visualize individual patches if required
                if visualize_individual and patches:
                    top_patches = [patches[j] for j in top_patch_indices if j < len(patches)]
                    self.visualize_patch_overlay(pcd, top_patches, colors, i, cls, mode=mode)


                if visualize:
                    self.visualize_all([self.full_scene_pcd] + all_patches)


    def visualize_patch_overlay(self, pcd_unused, patches, colors, box_index, cls, mode='3d'):
        # Visualize patches in 2D or 3D based on mode
        if mode == '2d':
            self.visualize_2d(patches, box_index, cls)
        elif mode == '3d':
            self.visualize_3d(patches, colors, box_index, cls)

    def visualize_2d(self, patches, box_index, cls):
        """Visualize patches in 2D on the image"""
        depth_norm = (self.depth_img / np.max(self.depth_img) * 255).astype(np.uint8)
        overlay = np.stack([depth_norm]*3, axis=-1)

        color_map = plt.cm.get_cmap("Set1", len(patches))
        for k, patch in enumerate(patches):
            corners_3d = np.asarray(patch.get_box_points())
            corners_2d = []

            for pt in corners_3d:
                if pt[2] <= 0:
                    continue
                u = int((pt[0] * self.fx) / pt[2] + self.cx)
                v = int((pt[1] * self.fy) / pt[2] + self.cy)
                if 0 <= u < overlay.shape[1] and 0 <= v < overlay.shape[0]:
                    corners_2d.append((u, v))
            if len(corners_2d) >= 4:
                corners_2d = np.array(corners_2d, dtype=np.int32)
                hull = cv2.convexHull(corners_2d)
                color = tuple(np.array(color_map(k)[:3]) * 255)
                cv2.fillPoly(overlay, [hull], color=color)
                cv2.polylines(overlay, [hull], isClosed=True, color=(0, 0, 0), thickness=2)

        # Display the 2D overlay using matplotlib
        plt.figure(figsize=(8, 6))
        plt.imshow(overlay)
        plt.title(f"Detection {box_index} - {cls}")
        plt.axis('off')  # Turn off axis labels
        plt.show()

    def visualize_3d(self, patches, colors, box_index, cls):
        """Visualize patches in 3D"""
        geometries = []
        pcd = self.full_scene_pcd
        geometries.append(pcd)

        for k, patch in enumerate(patches):
            mesh_box = o3d.geometry.LineSet.create_from_oriented_bounding_box(patch)
            mesh_box.paint_uniform_color(colors[k % len(colors)][:3])
            geometries.append(mesh_box)

        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        geometries.append(coord)

        o3d.visualization.draw_geometries(
            geometries,
            window_name=f"Detection {box_index} - {cls}",
            height=600
        )


    def colorize_pointcloud_from_image(self, pcd):
        fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy
        img_h, img_w = self.color_img.shape[:2]

        points = np.asarray(pcd.points)
        colors = []

        for pt in points:
            if pt[2] <= 0:
                colors.append([0, 0, 0])
                continue

            u = int((pt[0] * fx) / pt[2] + cx)
            v = int((pt[1] * fy) / pt[2] + cy)

            if 0 <= u < img_w and 0 <= v < img_h:
                color = self.color_img[v, u, :] / 255.0
            else:
                color = [0, 0, 0]

            colors.append(color)

        pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
        return pcd

    def filter_points_on_plane(self, pcd, plane_model, threshold=0.005):
        """
        Filter points from the point cloud that are within a distance to the plane.
        plane_model: [a, b, c, d] from ax + by + cz + d = 0
        threshold: max distance in meters
        """
        a, b, c, d = plane_model
        points = np.asarray(pcd.points)
        distances = np.abs((a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d)) / np.linalg.norm([a, b, c])
        mask = distances < threshold
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(points[mask])
        return filtered_pcd

    def visualize_colored_surfaces(self):
        """
        Visualize all cleaned and colorized planar surfaces.
        """
        geometries = []
        for surface in self.cleaned_box_surfaces:
            colored = self.colorize_pointcloud_from_image(surface)
            geometries.append(colored)
            # mesh = self.pointcloud_to_surface_mesh(surface)
            # mesh.paint_uniform_color([0.6, 0.8, 1.0])  # Light blue for visibility
            # geometries.append(mesh)

        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        geometries.append(coord)

        o3d.visualization.draw_geometries(geometries, window_name="Cleaned Colored Surfaces")
        # o3d.visualization.draw_geometries(geometries, window_name="Surface Meshes")


    def get_results(self):
        return list(zip(self.per_box_pcds, self.per_box_patches))

    def get_cleaned_surfaces(self):
        return self.cleaned_box_surfaces
