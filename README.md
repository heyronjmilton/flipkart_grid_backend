# Vision-Based Smart System for Quality Assessment

This project provides a FastAPI-based backend application for object detection, expiry detection, and fruit/vegetable classification using YOLO models. The application includes WebSocket endpoints for real-time detection and image processing and REST APIs for additional operations.

## Features

- Object detection for identifying products.
- Expiry detection for recognizing and saving expiration details.
- Fruit and vegetable classification with tracking capabilities.
- WebSocket endpoints for real-time video feed processing.
- Integration with YOLO models for high-performance inference.
- Configurable confidence thresholds for detection models.

## Installation

### Prerequisites

- Python 3.8 or later.
- GPU with CUDA support (recommended for better performance).
- Installed `venv` for virtual environment management.
- Installed [ultralytics](https://github.com/ultralytics/ultralytics) library for YOLO.

### Steps

1. Clone this repository:

   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   now install torch with CUDA

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

   now create a .env file with the field

   ```bash
   gemini_key=<api-key-here>
   ```

4. Place your YOLO models in the `model/` directory:

   - `object_detection.pt`
   - `expiry.pt`
   - `fruit.pt`

5. Run the application:
   ```bash
   uvicorn main:app --reload
   ```

## Endpoints

### WebSocket Endpoints

#### `/ws/camera_feed_expiry`

- Processes real-time video feed for packed product object and expiry detection.
- Input: Base64-encoded image frames.
- Output: Annotated frames with detected objects/expiry details.

#### `/ws/packed_products_expiry`

- Provides real-time updates on detected expiry details.

#### `/ws/camera_feed_fruit`

- Processes real-time video feed for fruit/vegetable detection.
- Input: Base64-encoded image frames.
- Output: Annotated frames with detected fruits/vegetables.

#### `/ws/fruits`

- Provides real-time updates on detected fruit/vegetable counts.

### REST Endpoints

#### `The REST Endpoints are documented in the /docs route of the server`

## Configuration

- Modify confidence thresholds for models in the script:
  ```python
  obj_conf = 0.5
  expiry_conf = 0.5
  fruit_conf = 0.5
  ```

## Folder Structure

```

├── reports
├── VIDEO_UPLOADS
├── details
├── data
|
├── model
│   ├── object_detection.pt
│   ├── expiry.pt
│   └── fruit.pt
├── utils
│   ├── image_process.py
│   ├── gemini_image.py
│   ├── handlelist.py
│   ├── handlereports.py
│   └── handleuploads.py
|
├── file_checker.py
├── main.py
├── requirements.txt
└── README.md
```

## Utilities

The `utils` directory contains helper scripts:

- `image_process.py`: For processing and saving images.
- `handlelist.py`: Manages lists and JSON operations.
- `handlereports.py`: Handles report generation in Excel format.
- `handleuploads.py`: Processes uploaded files.

## Notes

- Ensure that `file_checker.py` exists in the root directory for background file-checking functionality.
