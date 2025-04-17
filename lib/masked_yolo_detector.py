import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO


class MaskedYOLODetector:
    """
    A wrapper for YOLO detection that filters detections using a depth-based mask.

    Attributes:
        color_image (np.ndarray): The input BGR color image.
        depth_image (np.ndarray): The input depth image.
        model (YOLO): Loaded YOLO model from ultralytics.
        min_depth (int): Minimum valid depth threshold (in depth units).
        max_depth (int): Maximum valid depth threshold.
        mask (np.ndarray): Binary mask of region of interest.
        filtered_boxes (list): Final boxes that are within the mask and size constraints.
        min_box_area (int): Minimum allowed bounding box area.
        max_box_area (int): Maximum allowed bounding box area.
    """

    def __init__(self, color_image, depth_image, model_path,
                 min_depth=1390, max_depth=1766,
                 min_box_area=10, max_box_area=10000):
        """
        Initialize the detector and load the model.

        Args:
            color_image (np.ndarray): Color image (BGR).
            depth_image (np.ndarray): Corresponding depth image (single-channel).
            model_path (str): Path to the YOLO model (.pt).
            min_depth (int): Minimum depth to be considered valid.
            max_depth (int): Maximum depth to be considered valid.
            min_box_area (int): Minimum bounding box area to be valid.
            max_box_area (int): Maximum bounding box area to be valid.
        """
        self.color_image = color_image
        self.depth_image = depth_image
        self.model = YOLO(model_path)
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.mask = None
        self.filtered_boxes = []
        self.min_box_area = min_box_area
        self.max_box_area = max_box_area

    def set_box_area(self, min_area, max_area):
        """
        Update bounding box area constraints.

        Args:
            min_area (int): Minimum bounding box area.
            max_area (int): Maximum bounding box area.
        """
        self.min_box_area = min_area
        self.max_box_area = max_area

    def create_mask(self):
        """
        Generate a binary mask using depth filtering and morphological operations.
        Keeps the largest connected region to reduce false detections.
        """
        depth_mask = ((self.depth_image > self.min_depth) & (self.depth_image < self.max_depth)).astype(np.uint8) * 255

        if depth_mask.ndim == 3:
            depth_mask = cv.cvtColor(depth_mask, cv.COLOR_BGR2GRAY)
        if depth_mask.shape != self.color_image.shape[:2]:
            depth_mask = cv.resize(depth_mask, (self.color_image.shape[1], self.color_image.shape[0]))

        # Apply morphological operations to clean up the mask
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (30, 30))
        depth_mask = cv.morphologyEx(depth_mask, cv.MORPH_CLOSE, kernel)
        depth_mask = cv.dilate(depth_mask, kernel, iterations=1)

        # Find the largest connected component
        num_labels, labels, stats, _ = cv.connectedComponentsWithStats(depth_mask, connectivity=8)
        largest_label = 1 + np.argmax(stats[1:, cv.CC_STAT_AREA])
        self.mask = (labels == largest_label).astype(np.uint8) * 255

    def visualize_input(self):
        """
        Display the original color image, depth image, and the masked region using matplotlib.
        Used for debugging and inspection.
        """
        depth_display = cv.normalize(self.depth_image, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        masked_color = cv.bitwise_and(self.color_image, self.color_image, mask=self.mask)
        masked_color = cv.cvtColor(masked_color, cv.COLOR_BGR2RGB)

        plt.figure(figsize=(12, 6))
        plt.subplot(131)
        plt.imshow(cv.cvtColor(self.color_image, cv.COLOR_BGR2RGB))
        plt.title('Original Color Image')
        plt.axis('off')

        plt.subplot(132)
        plt.imshow(depth_display, cmap='plasma')
        plt.title('Depth Image')
        plt.axis('off')

        plt.subplot(133)
        plt.imshow(masked_color)
        plt.title('Masked Region')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def run_detection(self, visualize=False):
        """
        Run the YOLO model and filter bounding boxes using the mask and area limits.

        Args:
            visualize (bool): If True, displays detection results using matplotlib.

        Returns:
            list: Filtered boxes in the form (class_name, confidence, (x1, y1, x2, y2)).
        """
        results = self.model(self.color_image)
        result_img = self.color_image.copy()
        boxes = results[0].boxes
        total_boxes = len(boxes)

        filtered_count = 0
        for box in boxes:
            cls = self.model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

            if 0 <= center_y < self.mask.shape[0] and 0 <= center_x < self.mask.shape[1]:
                is_in_mask = self.mask[center_y, center_x] > 0
            else:
                is_in_mask = False

            box_area = (x2 - x1) * (y2 - y1)
            if box_area < self.min_box_area or box_area > self.max_box_area or not is_in_mask:
                continue

            filtered_count += 1
            self.filtered_boxes.append((cls, conf, (x1, y1, x2, y2)))

            if visualize:
                color = (0, 255, 0)
                label = f"{cls} {conf:.2f}"
                cv.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
                text_size = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv.rectangle(result_img, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
                cv.putText(result_img, label, (x1, y1 - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if visualize:
            plt.figure(figsize=(16, 10))
            plt.subplot(1, 2, 1)
            plt.imshow(cv.cvtColor(self.color_image, cv.COLOR_BGR2RGB))
            plt.title('Original Image')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(cv.cvtColor(result_img, cv.COLOR_BGR2RGB))
            plt.title(f'Detections: {filtered_count}/{total_boxes} inside mask')
            plt.axis('off')
            plt.tight_layout()
            plt.show()

        return self.filtered_boxes

    def print_box_details(self):
        """
        Print the details of the filtered boxes and their mask overlap percentage.
        """
        print(f"Found {len(self.filtered_boxes)} boxes inside the mask:")
        for cls, conf, (x1, y1, x2, y2) in self.filtered_boxes:
            print(f"Class: {cls}, Confidence: {conf:.2f}, Box: ({x1}, {y1}, {x2}, {y2})")

        print("\nDetailed mask overlap analysis:")
        for cls, conf, (x1, y1, x2, y2) in self.filtered_boxes:
            box_mask = self.mask[y1:y2, x1:x2]
            area = (y2 - y1) * (x2 - x1)
            overlap = np.count_nonzero(box_mask) / area if area > 0 else 0
            print(f"{cls}: {conf:.2f} - Mask overlap: {overlap:.2%}")
