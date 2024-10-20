import cv2
import os
from utils.gemini_image import process_image
import json

def append_to_json_file(expiry, mfg, batch_no,object_name):
    # Check if the file exists
    if os.path.exists("data/expiry_details.json"):
        # Read the existing data
        with open("data/expiry_details.json", 'r') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = []  # If file is empty or has invalid JSON
    else:
        data = []

    # Create a new entry
    new_entry = {
        'expiry': expiry,
        'mfg': mfg,
        'batch_no': batch_no,
        'object_name': object_name
    }

    # Append new entry
    data.append(new_entry)

    # Write back to the file
    with open("data/expiry_details.json", 'w') as file:
        json.dump(data, file, indent=4)


def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

def save_expiry_image(image, x1, y1, x2, y2, object_name):
    cropped_image = image[y1:y2, x1:x2]
    object_folder = "details"
    file_path = os.path.join(object_folder, f"{object_name}.jpg")
    cv2.imwrite(file_path, cropped_image)
    return file_path
