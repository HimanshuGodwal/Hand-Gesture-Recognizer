import os
import math

import cv2
import numpy as np


# Load all folder names
def load_class_names(data_dir):

    labels = []

    for entry in os.listdir(data_dir):

        full_path = os.path.join(data_dir, entry)

        if os.path.isdir(full_path):
            labels.append(entry)

    labels.sort()

    return labels


# Save class names into text file
def save_class_names(labels, path):

    with open(path, "w", encoding="utf-8") as file:

        for label in labels:
            file.write(f"{label}\n")


# Create white background hand image
def make_hand_canvas(image, bbox, img_size=224, offset=20):

    x, y, w, h = bbox

    # Add extra space around hand
    x1 = max(x - offset, 0)
    y1 = max(y - offset, 0)

    x2 = min(x + w + offset, image.shape[1])
    y2 = min(y + h + offset, image.shape[0])

    # Stop if box is invalid
    if x2 <= x1 or y2 <= y1:
        return None

    # Crop hand area
    img_crop = image[y1:y2, x1:x2]

    # Stop if crop failed
    if img_crop.size == 0:
        return None

    # Create white square image
    img_white = np.ones(
        (img_size, img_size, 3),
        np.uint8
    ) * 255

    crop_h, crop_w = img_crop.shape[:2]

    aspect_ratio = crop_h / crop_w

    # If hand is tall
    if aspect_ratio > 1:

        scale = img_size / crop_h

        resized_w = math.ceil(scale * crop_w)

        resized_w = max(1, resized_w)

        resized_w = min(img_size, resized_w)

        img_resize = cv2.resize(
            img_crop,
            (resized_w, img_size)
        )

        w_gap = math.ceil(
            (img_size - resized_w) / 2
        )

        img_white[:, w_gap:w_gap + resized_w] = img_resize

    # If hand is wide
    else:

        scale = img_size / crop_w

        resized_h = math.ceil(scale * crop_h)

        resized_h = max(1, resized_h)

        resized_h = min(img_size, resized_h)

        img_resize = cv2.resize(
            img_crop,
            (img_size, resized_h)
        )

        h_gap = math.ceil(
            (img_size - resized_h) / 2
        )

        img_white[h_gap:h_gap + resized_h, :] = img_resize

    return img_white