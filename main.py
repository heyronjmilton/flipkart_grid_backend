from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import base64
import cv2
import numpy as np

import torch
from ultralytics import YOLO

from utils.image_process import resize_with_aspect_ratio


device = torch.device("cuda")

object_detection_model = YOLO("model\object_detection.pt")
object_detection_model.info()

object_detection_model = object_detection_model.to(device)

app = FastAPI()

# Allow CORS for your frontend application (if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws/camera_feed")
async def websocket_camera_feed(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection established")

    try:
        while True:
            # Wait for the client to send an image
            image_data = await websocket.receive_text()

            # Extract the base64 string from the data URL
            header, encoded = image_data.split(',', 1)
            # Decode the image
            data = base64.b64decode(encoded)
    
            # Convert to a numpy array and decode the image
            img_array = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            resized_frame = resize_with_aspect_ratio(img, width=640)

            results_object_detection = object_detection_model(resized_frame)
            result_object_detection = results_object_detection[0]

            image_with_boxes_object = result_object_detection.plot()

            # Display the image using OpenCV
            cv2.imshow("Camera Feed", resized_frame)
            cv2.imshow("Inferneced Camera Feed", image_with_boxes_object)
            cv2.waitKey(1)  # Display the image for 1 ms

            # Encode the image to base64 to send it back
            _, buffer = cv2.imencode('.jpg', image_with_boxes_object)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            await websocket.send_text(f"data:image/jpeg;base64,{jpg_as_text}")  # Send the image back

    except WebSocketDisconnect:
        print("WebSocket connection closed.")
        cv2.destroyAllWindows()  # Close the preview window when the connection is closed

@app.get("/")
def home():
    return {"message" : "The server is up and running"}