import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO


class MaskedYOLODetector:
    def __init__(self, color_image, depth_image, model_path,
                 min_depth=1390, max_depth=1766,
                 min_box_area=10, max_box_area=10000):
        """
        Initialize the Masked YOLO Detector.
        :param color_image: The color image (BGR format).
        :param depth_image: The depth image (single channel).
        :param model_path: Path to the YOLO model.
        :param min_depth: Minimum depth for filtering out intersting  area
        :param max_depth: Maximum depth for filtering out intersting area
        :param min_box_area: Minimum area of the bounding box to consider
        :param max_box_area: Maximum area of the bounding box to consider
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
        Set the minimum and maximum area for the bounding boxes.
        :param min_area: Minimum area of the bounding box to consider
        :param max_area: Maximum area of the bounding box to consider
        """
        self.min_box_area = min_area
        self.max_box_area = max_area

    def create_mask(self):
        """
        Create a mask based on the depth image to filter out regions of interest.
        The mask is created by thresholding the depth image and applying morphological operations.
        The largest connected component is retained as the mask.
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
        Run the YOLO detection on the color image and filter the results based on the mask.
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
        Print the detals of the filtered boxes
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
