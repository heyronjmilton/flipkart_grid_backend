from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import base64
import cv2
import numpy as np
import asyncio
import random
import json
import torch
import time
from ultralytics import YOLO
from collections import defaultdict
from ultralytics.utils.plotting import Annotator, colors
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


item_objects = [
    {"name": "Box of Cereal", "type": "packaged", "exp": "2025-01-01"},
    {"name": "Snack Bar", "type": "packaged", "exp": "2024-07-15"},
]

item_fruits = [
     {"name": "Apple", "type": "fruit", "quality": "Good"},
    {"name": "Banana", "type": "fruit", "quality": "Ripe"},
]

# these are for counting functionality
out = cv2.VideoWriter("object-tracking.avi", cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 640))
max_inactive_time = 1.0
track_history = defaultdict(lambda: {'last_seen': time.time(), 'box': None, 'confidence': 0})
detected_objects_list = []


@app.websocket("/ws/camera_feed_object")
async def websocket_camera_feed_object(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection established for object detection")

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

@app.websocket("/ws/objects")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Send item updates to the connected client
            try:
                await websocket.send_text(json.dumps(item_objects))  # Convert items to JSON string
            except Exception as e:
                print(f"Error sending message: {e}")
                break  # Break the loop if there is an error in sending
            
            await asyncio.sleep(1)  # Wait for 5 seconds before sending again
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()


@app.websocket("/ws/camera_feed_fruit")
async def websocket_camera_feed_fruit(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection established for fruit detection")

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

@app.websocket("/ws/fruits")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Send item updates to the connected client
            try:
                await websocket.send_text(json.dumps(item_fruits))  # Convert items to JSON string
            except Exception as e:
                print(f"Error sending message: {e}")
                break  # Break the loop if there is an error in sending
            
            await asyncio.sleep(1)  
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

@app.websocket("/ws/camera_feed_count")
async def websocket_camera_feed_count(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection established for count detection")

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

            resized_frame = cv2.resize(img, (640, 640))

            results = object_detection_model.track(resized_frame, verbose=False, persist=True)
            current_time = time.time()
            annotator = Annotator(resized_frame, line_width=2)
            
            global detected_objects_list
            detected_objects  = []
            detected_objects_list = []
            if results and results[0].boxes:  # Ensure there are boxes detected
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                classes = results[0].boxes.cls.int().cpu().tolist()
                track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else [None] * len(boxes)

                # Update track history with current frame data
                for box, track_id, confidence, cls in zip(boxes, track_ids, confidences, classes):
                    if confidence > 0.65:
                        x1, y1, x2, y2 = map(int, box)
                        obj_name = object_detection_model.names[cls]

                        # Store the box and confidence for smoothing
                        track_history[track_id]['last_seen'] = current_time
                        track_history[track_id]['box'] = (x1, y1, x2, y2)
                        track_history[track_id]['confidence'] = confidence
                        track_history[track_id]['name'] = obj_name
            else:
                print("No boxes detected in this frame.")

            # Draw boxes from history to prevent flicker
            for track_id, data in track_history.items():
                # If the object has been inactive for too long, skip it
                if current_time - data['last_seen'] > max_inactive_time:
                    continue
                
                # Use the last known position to draw the box
                if data['box'] is not None:  # Check if the box exists
                    x1, y1, x2, y2 = data['box']
                    label = f"{data['name']} (ID: {track_id}) {data['confidence']:.2f}"
                    
                    # Ensure track_id is not None before passing to colors
                    if track_id is not None:
                        color = colors(track_id, True)
                    else:
                        color = (255, 0, 0)  # Default color if track_id is None
                    
                    annotator.box_label((x1, y1, x2, y2), label, color=color)
                    detected_objects.append((data['name'], track_id, data['confidence']))

            # Get the annotated frame for display
            annotated_frame = annotator.result()

            # Write the annotated frame to the output video
            out.write(annotated_frame)


            if detected_objects:
                print("Detected objects in this frame:")
                for obj_name, track_id, confidence in detected_objects:
                    print(f"Object: {obj_name}, Track ID: {track_id}, Confidence: {confidence:.2f}")
                    detected_objects_list.append({
                        "object": obj_name,
                        "track_id": track_id,
                        "confidence": round(float(confidence),2)  # rounding to 2 decimal places
                    })
            else:
                print("No objects detected with confidence > 0.65 in this frame.")


            # Display the image using OpenCV
            cv2.imshow("Camera Feed", resized_frame)
            cv2.imshow("Tracking Camera Feed", annotated_frame)
            cv2.waitKey(1)  # Display the image for 1 ms

            # Encode the image to base64 to send it back
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            await websocket.send_text(f"data:image/jpeg;base64,{jpg_as_text}")  # Send the image back

    except WebSocketDisconnect:
        print("WebSocket connection closed.")
        cv2.destroyAllWindows()  # Close the preview window when the connection is closed


@app.websocket("/ws/count")
async def websocket_endpoint(websocket: WebSocket):
    global detected_objects_list
    await websocket.accept()
    try:
        while True:
            # Send item updates to the connected client
            try:
                await websocket.send_text(json.dumps(detected_objects_list))  # Convert items to JSON string
            except Exception as e:
                print(f"Error sending message: {e}")
                break  # Break the loop if there is an error in sending
            
            await asyncio.sleep(1)  
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()


@app.get("/")
def home():
    return {"message" : "The server is up and running"}

