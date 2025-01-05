from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

import base64
import cv2
import numpy as np
import asyncio, json, torch, time, os, subprocess, threading
from collections import defaultdict, deque, Counter
import ast

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

from utils.image_process import save_expiry_image
from utils.handlelist import make_object_final, clear_list
from utils.handlereports import save_expiry_details_to_excel, save_fruit_details_to_excel
from utils.handleuploads import handle_file_upload
from utils.gemini_processor import process_image

import hashlib
import shutil
import boto3
from botocore.exceptions import NoCredentialsError
from dotenv import load_dotenv

device = torch.device("cuda")

object_detection_model = YOLO("model/object_detection.pt")
expiry_detection_model = YOLO('model/expiry.pt')
fruit_detection_model = YOLO('model/fruit.pt')

fruit_detection_model.info()
object_detection_model.info()
expiry_detection_model.info()

object_detection_model = object_detection_model.to(device)
expiry_detection_model = expiry_detection_model.to(device)
fruit_detection_model = fruit_detection_model.to(device)


app = FastAPI()

# Allow CORS for your frontend application (if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
device_id = os.getenv('DEVICE_ID')

obj_conf = 0.5 #model confidence variables
expiry_conf = 0.5
fruit_conf = 0.5


process = None          #for the working of the file checker
process_lock = threading.Lock()
with process_lock:
        # If a process is already running, kill it
        if process is not None and process.poll() is None:
            process.kill()
            process = None  # Clear the process reference
        
        # Start a new subprocess
        if os.name == 'nt':  # For Windows
            process = subprocess.Popen(['venv\\Scripts\\python.exe', 'file_checker.py'])
        else:  # For Linux/macOS
            process = subprocess.Popen(['venv/bin/python', 'file_checker.py'])
        print(f"Started subprocess with PID: {process.pid}")

        print({"message": "Process restarted successfully", "pid": process.pid})

buffer_list = []   #product detection and expiry detection variables
name_detection = False
product_name = None
in_sensor = False
out_sensor = False
product_dict = {}

fruit_veggie_buffer_length = 200
fruit_veggie_buffer = deque(maxlen=fruit_veggie_buffer_length)
fruit_veggie_final_dict = {}
current_fruit_veggie_dict={}
prev_fruit_veggie_count={}
total_fruit_veggie_count = 0
current_fruit_veggie_count = 0
realtime_fruit_veggie_dict = {}
fruitFlag = True
detection_progress = 0

gpt_response = ""

frame_queue = deque(maxlen=1) #queue to get only the latest frames

clear_list("expiry_details.json")

# these are for counting functionality
out = cv2.VideoWriter("object-tracking.avi", cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 640))
max_inactive_time = 1.0
track_history = defaultdict(lambda: {'last_seen': time.time(), 'box': None, 'confidence': 0})
detected_objects_list = []
detected_fruits_list = []

report_generated = False


def Most_Common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]

def upload_to_s3(bucket_name, file_name, object_name=None):
    """
    Upload a file to an S3 bucket.

    :param bucket_name: Bucket to upload to
    :param file_name: File to upload
    :param object_name: S3 object name. If not specified, file_name is used.
    :return: True if file was uploaded, else False
    """
    # Use the file name if no object name is provided
    if object_name is None:
        object_name = file_name

    # Create an S3 client
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_REGION')
    )

    try:
        # Upload the file
        s3_client.upload_file(file_name, bucket_name, object_name)
        print(f"File {file_name} uploaded to {bucket_name}/{object_name}")
        return True
    except FileNotFoundError:
        print(f"File {file_name} not found.")
    except NoCredentialsError:
        print("Credentials not available.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return False



async def process_object_detection(latest_frame):
    global buffer_list, name_detection, product_name, detection_progress
    
    updated_frame = latest_frame.copy()
    height, width = updated_frame.shape[:2]  # Get image dimensions
    results_object_detection = object_detection_model(updated_frame, verbose=False)
    yolo_annotations = []  # List to store YOLO format annotations
    
    for box in results_object_detection[0].boxes:
        confidence = box.conf.item()
        if confidence > obj_conf:
            name = results_object_detection[0].names[int(box.cls)]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Calculate YOLO format coordinates (normalized)
            x_center = ((x1 + x2) / 2) / width
            y_center = ((y1 + y2) / 2) / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height
            
            # Store YOLO format annotation: class_id x_center y_center width height
            class_id = int(box.cls)
            yolo_annotation = f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
            yolo_annotations.append(yolo_annotation)
            
            # Original visualization code
            label = f"{name} {confidence:.2f}"
            print(f"NAME : {name}")
            buffer_list.append(name)
            detection_progress = (len(buffer_list) / 25) * 100
            cv2.rectangle(updated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(updated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            if len(buffer_list) == 25:
                print("buffer list full")
                product_name = Most_Common(buffer_list)
                print(f"PRODUCT NAME : {product_name}")
                name_detection = False
        else:
            label = f"NONE {confidence:.2f}"
            print(f"NULL NAME : {label}")
    
    return updated_frame, yolo_annotations

async def process_expiry_detection(resized_frame):
    global buffer_list, name_detection, product_name
    updated_frame = resized_frame.copy()
    results_expiry_detection = expiry_detection_model(updated_frame, verbose=False)
    for box in results_expiry_detection[0].boxes:
        confidence = box.conf.item()
        if confidence > expiry_conf:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            save_expiry_image(updated_frame, x1, y1, x2, y2, product_name)
            cv2.rectangle(updated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(updated_frame, f"Expiry {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return updated_frame

async def process_fruit_detection(resized_frame):
    results_fruit_detection = fruit_detection_model.track(resized_frame, verbose=False, persist=True)

    inferenced_frame = resized_frame.copy()
    class_name_list = []
    for box in results_fruit_detection[0].boxes:
        confidence = box.conf.item()
        if confidence > fruit_conf:
            name = results_fruit_detection[0].names[int(box.cls)]
            class_name_list.append(name)
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(inferenced_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(inferenced_frame, f"{name} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    class_name_dict = dict(Counter(class_name_list).most_common())

    return inferenced_frame, class_name_dict


@app.websocket("/ws/camera_feed_expiry")
async def websocket_camera_feed_packed_products(websocket: WebSocket):
    
    await websocket.accept()
    print("WebSocket connection established for object detection")

    global in_sensor, buffer_list, product_name, name_detection, report_generated, detection_progress
    report_generated = False

    try:
        while True:
            # Wait for the client to send an image
            image_data = await websocket.receive_text()
            header, encoded = image_data.split(',', 1)
            data = base64.b64decode(encoded)

            frame_queue.clear()
            frame_queue.append(data)
            latest_data = frame_queue[0]
            
            img_array = np.frombuffer(latest_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            latest_frame = cv2.resize(img, (640, 640))
            annotation = None
            if in_sensor:
                if name_detection:
                    updated_frame , annotation = await process_object_detection(latest_frame)
                    
                    if annotation:
                        # Convert the frame to bytes
                        annotated_frame_bytes = updated_frame.tobytes()

                        # Hash the bytes using SHA-256
                        annotated_frame_name = hashlib.sha256(annotated_frame_bytes).hexdigest()

                        # Convert the frame to bytes
                        frame_bytes = latest_frame.tobytes()

                        # Hash the bytes using SHA-256
                        frame_name = hashlib.sha256(frame_bytes).hexdigest()

                        os.makedirs("temp/annoated_frame", exist_ok=True)  # Create the directory if it doesn't exist
                        os.makedirs("temp/images/", exist_ok=True)  # Create the directory if it doesn't exist
                        os.makedirs("temp/labels/", exist_ok=True)  # Create the directory if it doesn't exist

                        cv2.imwrite(f"temp/annoated_frame/{annotated_frame_name}.png",updated_frame)
                        cv2.imwrite(f"temp/images/{frame_name}.png",latest_frame)
                        with open(f"temp/labels/{frame_name}.txt", 'w') as file:
                            file.write( "\n".join(annotation))
                        

                        annotation = None
                else:
                    updated_frame = await process_expiry_detection(latest_frame)
            else:
                updated_frame = latest_frame


            if(not in_sensor) :
                buffer_list = []
                if product_name != None :
                    make_object_final(product_name,"expiry_details.json")
                product_name = None
                name_detection = True
                # print("not in active state")

            
            # cv2.imshow("Camera Feed", resized_frame)
            # cv2.imshow("Object and expiry detection", resized_frame)
            # cv2.waitKey(1)  # Display the image for 1 ms

            # Encode the image to base64 to send it back
            _, buffer = cv2.imencode('.jpg', updated_frame)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            await websocket.send_text(f"data:image/jpeg;base64,{jpg_as_text}")  # Send the image back

    except WebSocketDisconnect:
        print("Packed Items WebSocket connection closed.")
        cv2.destroyAllWindows()  # Close the preview window when the connection is closed


@app.websocket("/ws/packed_products_expiry")
async def packed_products_expiry(websocket: WebSocket):
    global product_name, name_detection, report_generated
    await websocket.accept()
    report_generated = False
    try:
        while True:
            if os.path.exists("data/expiry_details.json"):
            # Send item updates to the connected client
                with open("data/expiry_details.json", 'r') as file:
                    data = json.load(file)
                try:
                    data_to_send = {
                        "details" : data,
                        "count" : len(data),
                        "product_name" : product_name,
                        "name_detection" : name_detection,
                        "report_generated" : report_generated,
                        "in_sensor" : in_sensor,
                        "detection_progress" : detection_progress
                    }
                    await websocket.send_text(json.dumps(data_to_send))  # Convert items to JSON string
                except Exception as e:
                    print(f"Error sending message: {e}")
                    break  # Break the loop if there is an error in sending
                
                await asyncio.sleep(1)  # Wait for 5 seconds before sending again
            else :
                 with open("data/expiry_details.json", 'w') as file:
                    data = []
                    json.dump(data, file, indent=4)
            
    except Exception as e:
        print(f"Packed Item WebSocket error: {e}")
    # finally:
    #     await websocket.close()


@app.websocket("/ws/camera_feed_fruit")
async def websocket_camera_feed_fruit(websocket: WebSocket):
    await websocket.accept()
    global fruit_veggie_final_dict,current_fruit_veggie_dict,prev_fruit_veggie_count, total_fruit_veggie_count, current_fruit_veggie_count, realtime_fruit_veggie_dict, report_generated, fruitFlag
    report_generated = False
    print("WebSocket connection established for fruit detection")

    try:
        while True:
            # Wait for the client to send an image
            image_data = await websocket.receive_text()
            header, encoded = image_data.split(',', 1)
            data = base64.b64decode(encoded)

            frame_queue.clear()
            frame_queue.append(data)
            latest_data = frame_queue[0]
            
            img_array = np.frombuffer(latest_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            latest_frame = cv2.resize(img, (640, 640))
            
            if in_sensor:
                inferenced_frame, class_name_dict = await process_fruit_detection(latest_frame)
                realtime_fruit_veggie_dict = Counter(class_name_dict)
                class_name_dict_string = str(class_name_dict)
                
                fruit_veggie_buffer.append(class_name_dict_string)
                current_fruit_veggie_dict=ast.literal_eval(Counter(fruit_veggie_buffer).most_common()[0][0])
                current_fruit_veggie_count = sum(realtime_fruit_veggie_dict.values())
                fruitFlag = True
                # print(f"current details {realtime_fruit_veggie_dict} current count : {current_fruit_veggie_count}")
            else :
                # print("inactive state")
                realtime_fruit_veggie_dict = {}
                current_fruit_veggie_count = 0
                inferenced_frame = latest_frame 
                if(fruitFlag):
                    fruit_veggie_final_dict=Counter(fruit_veggie_final_dict)+Counter(current_fruit_veggie_dict)
                    fruit_veggie_buffer.clear() 
                    total_fruit_veggie_count = sum(fruit_veggie_final_dict.values())  
                    fruit_veggie_final_dict=dict(fruit_veggie_final_dict)
                    print(fruit_veggie_final_dict)   
                    prev_fruit_veggie_count=current_fruit_veggie_dict
                    fruitFlag = False 

                    # Encode the image to base64 to send it back
            _, buffer = cv2.imencode('.jpg', inferenced_frame)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            await websocket.send_text(f"data:image/jpeg;base64,{jpg_as_text}")  # Send the image back

    except WebSocketDisconnect:
        print("WebSocket connection closed.")
        fruit_veggie_buffer.clear()
        cv2.destroyAllWindows()  # Close the preview window when the connection is closed

@app.websocket("/ws/fruits")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    global fruit_veggie_final_dict, current_fruit_veggie_dict, total_fruit_veggie_count, current_fruit_veggie_count
    try:
        while True:
            # Send item updates to the connected client
            try:
                data_to_send = {
                        "details" : fruit_veggie_final_dict,
                        "current_details" : realtime_fruit_veggie_dict,
                        "count" : total_fruit_veggie_count,
                        "current_count" : current_fruit_veggie_count,
                        "in_sensor" : in_sensor,
                        "report_generated" : report_generated
                    }
                await websocket.send_text(json.dumps(data_to_send))  # Convert items to JSON string
            except Exception as e:
                print(f"Error sending message: {e}")
                break  # Break the loop if there is an error in sending
            
            await asyncio.sleep(1)  
    except Exception as e:
        print(f"WebSocket error: {e}")
    # finally:
    #     await websocket.close()


@app.get("/reset-detection")
def resetDetection():
    global buffer_list, name_detection, product_name, report_generated, fruit_veggie_buffer, detection_progress, gpt_response

    gpt_response = []
    buffer_list = []
    detection_progress = 0
    name_detection = True
    product_name = None
    report_generated = False
    fruit_veggie_buffer.clear()
    return {"msg" : "detected objected resetted"}


@app.get("/set-in-sensor")
async def setNameDetection(value : int):
    global in_sensor, buffer_list, name_detection, product_name, report_generated, fruit_veggie_buffer, fruitFlag, process
    if(int(value) == 1) :
        if in_sensor == False :
            in_sensor = True
            product_name = None
            buffer_list = []
            name_detection = True
            fruit_veggie_buffer.clear()

        
    elif(int(value) == 0) :
        if in_sensor == True :
            in_sensor = False
            name_detection = True
            buffer_list = []
            fruit_veggie_buffer.clear()
            fruitFlag = True
            with process_lock:
                if process is not None and process.poll() is None:
                    process.kill()
                    process = None  # Clear the process reference
            
                if os.name == 'nt':  # For Windows
                    process = subprocess.Popen(['venv\\Scripts\\python.exe', 'file_checker.py'])
                else:  # For Linux/macOS
                    process = subprocess.Popen(['venv/bin/python', 'file_checker.py'])
                print(f"Started subprocess with PID: {process.pid}")
            
    report_generated = False
    return {"in_sensor" : in_sensor, "name_detection" : name_detection, "product_name" : product_name }


@app.get("/start-file-checker")
async def file_checker():
    global process
    with process_lock:
        # If a process is already running, kill it
        if process is not None and process.poll() is None:
            process.kill()
            process = None  # Clear the process reference
        
        # Start a new subprocess
        if os.name == 'nt':  # For Windows
            process = subprocess.Popen(['venv\\Scripts\\python.exe', 'file_checker.py'])
        else:  # For Linux/macOS
            process = subprocess.Popen(['venv/bin/python', 'file_checker.py'])
        print(f"Started subprocess with PID: {process.pid}")
        return {"message": "Process restarted successfully", "pid": process.pid}

@app.get("/stop-file-checker")
async def stop_process():
    global process
    with process_lock:
        if process is None or process.poll() is not None:
            raise HTTPException(status_code=400, detail="No process is running.")
        
        # Force kill the subprocess
        process.kill()
        return {"message": "Process forcefully killed"}

@app.post("/finish-task")
async def finsihTask(batch_name:str, tasktype:str):

    global in_sensor, report_generated, fruit_veggie_final_dict, total_fruit_veggie_count
    reports_folder = "reports"

    if(tasktype == "packed") :
        print("PROCESSING ITEM DETECTION REPORT")
        with open(f"data/expiry_details.json", 'r') as file:
            data = json.load(file)
        save_expiry_details_to_excel(data,reports_folder,f"{batch_name}_expiry_details.xlsx")
        clear_list("expiry_details.json")

        try:
            folder_path = "temp"
            output_path = f"{batch_name}"

            # Create ZIP archive
            shutil.make_archive(output_path, 'zip', folder_path)

            # Delete the folder
            shutil.rmtree(folder_path)

            print(f"Zipped contents to '{output_path}' and deleted folder '{folder_path}'.")

        except Exception as e:
            print(f"An error occurred: {e}")

        BUCKET_NAME = "flipkart-reports"
        FILE_NAME = f"reports/{batch_name}_expiry_details.xlsx"
        OBJECT_NAME = f"{device_id}_{batch_name}_report.xlsx"  

        # Call the upload function
        upload_to_s3(BUCKET_NAME, FILE_NAME, OBJECT_NAME)


        BUCKET_NAME = "ziplogs-flipkart"
        FILE_NAME = f"{batch_name}.zip"
        OBJECT_NAME = f"{device_id}/_{batch_name}.zip"  # Optional, specify custom object name if needed

        # Call the upload function
        upload_to_s3(BUCKET_NAME, FILE_NAME, OBJECT_NAME)

        os.remove(FILE_NAME)

        print("PROCESSING COMPLETE")

    elif(tasktype == "fruit") :
        save_fruit_details_to_excel(fruit_veggie_final_dict, reports_folder, f"{batch_name}_fruit_details.xlsx")

        print("PROCESSING FRUIT DETECTION REPORT")
        fruit_veggie_final_dict = {}
        total_fruit_veggie_count = 0

    else:
        print(f"ERROR TASK TYPE : {tasktype}")
        report_generated = False
        return {"msg" : "invalid task details"}

    in_sensor = False
    report_generated = True
    return {"msg" : f"{tasktype} details saved"}

@app.get("/get-sensor-data")
def getSensorData():
    global in_sensor, out_sensor
    data = {
        "in_sensor" : in_sensor,
        "out_sensor" : out_sensor
    }
    return data

@app.get("/download-report")
async def download_xlsx(batch_name: str,tasktype: str):
    global report_generated
    FILES_FOLDER = "reports"
    # Ensure the requested file name ends with .xlsx
    if tasktype == "packed" :
        file_name = f"{batch_name}_expiry_details.xlsx"
        print(f"{file_name}")
    elif tasktype == "fruit" :
        file_name = f"{batch_name}_fruit_details.xlsx"
    else :
        raise HTTPException(status_code=404, detail=f"INVALID TASK TYPE")
    # Construct the full path to the file
    file_path = os.path.join(FILES_FOLDER, file_name)
    
    # Check if the file exists
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File '{file_name}' not found in the folder")
    
    # Return the .xlsx file as a response
    report_generated = False
    return FileResponse(
        file_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=file_name
    )


@app.post("/upload-data-video")
async def uploadDataVideo(file: UploadFile = File(...), class_name: str = Form(...),item_type: str = Form(...)):
    print(f"filename : {file.filename}")
    new_filename = f"{class_name}#{item_type}.mp4"
    file.filename = new_filename
    response_data = await handle_file_upload(file, class_name, item_type)
    return JSONResponse(
        content=response_data,
        status_code=200
    )


@app.websocket("/ws/camera_feed_gpt")
async def websocket_camera_feed_packed_products(websocket: WebSocket):
    
    await websocket.accept()
    print("WebSocket connection established for object detection")

    global gpt_response
    report_generated = False
    image_send = True
    try:
        while True:
            # Wait for the client to send an image
            image_data = await websocket.receive_text()
            header, encoded = image_data.split(',', 1)
            data = base64.b64decode(encoded)

            frame_queue.clear()
            frame_queue.append(data)
            latest_data = frame_queue[0]
            
            img_array = np.frombuffer(latest_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            latest_frame = cv2.resize(img, (640, 640))
            

            if in_sensor:
                # print("active state")
                image_send = False
                updated_frame = latest_frame
            else:
                updated_frame = latest_frame


            if(not in_sensor) :
                
                # print("not in active state")

                if(not image_send) :
                    gpt_response = process_image(img,"test")
                    # print("response in server",gpt_response)
                    image_send = True



            
           
            # Encode the image to base64 to send it back
            _, buffer = cv2.imencode('.jpg', updated_frame)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            await websocket.send_text(f"data:image/jpeg;base64,{jpg_as_text}")  # Send the image back

    except WebSocketDisconnect:
        print("Packed Items WebSocket connection closed.")
        cv2.destroyAllWindows()  # Close the preview window when the connection is closed


@app.websocket("/ws/gpt_response")
async def packed_products_expiry(websocket: WebSocket):
    global product_name, name_detection, report_generated
    await websocket.accept()
    report_generated = False
    try:
        while True:
            if os.path.exists("data/expiry_details.json"):
            # Send item updates to the connected client
                with open("data/expiry_details.json", 'r') as file:
                    data = json.load(file)
                try:
                    data_to_send = {
                        "msg" : gpt_response
                    }
                    await websocket.send_text(json.dumps(data_to_send))  # Convert items to JSON string
                except Exception as e:
                    print(f"Error sending message: {e}")
                    break  # Break the loop if there is an error in sending
                
                await asyncio.sleep(1)  
            else :
                 with open("data/expiry_details.json", 'w') as file:
                    data = []
                    json.dump(data, file, indent=4)
            
    except Exception as e:
        print(f"Packed Item WebSocket error: {e}")
    # finally:
    #     await websocket.close()


@app.get("/")
def home():
    return {"message" : "The server is up and running"}

