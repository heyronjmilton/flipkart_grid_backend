from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import base64
import cv2
import numpy as np

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

            # Display the image using OpenCV
            cv2.imshow("Camera Feed", img)
            cv2.waitKey(1)  # Display the image for 1 ms

    except WebSocketDisconnect:
        print("WebSocket connection closed. Closing the preview window.")
        cv2.destroyAllWindows()  # Close the preview window when the connection is closed

@app.post("/upload_feed")
async def upload_feed(request: Request):
    data = await request.json()
    image = data.get("image")
    
    if not image:
        return JSONResponse(status_code=422, content={"message": "Invalid image data."})

    # Extract the base64 string from the data URL
    header, encoded = image.split(',', 1)
    
    # Decode the image
    data = base64.b64decode(encoded)
    
    # Convert to a numpy array and decode the image
    img_array = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Display the image using OpenCV
    cv2.imshow("Camera Feed", img)
    cv2.waitKey(1)  # Display the image for 1 ms

    return {"status": "success", "message": "Image received"}
