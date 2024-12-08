import google.generativeai as genai
import cv2
import os
import tempfile
from dotenv import load_dotenv
import json
import requests

load_dotenv()

# Configure the API key
genai.configure(api_key=os.getenv('gemini_key'))

# Initialize the model
model = genai.GenerativeModel('gemini-1.5-flash')


# def append_to_json_file(expiry, mfg, batch_no,object_name):
#     # Check if the file exists
#     if os.path.exists("data/expiry_details.json"):
#         # Read the existing data
#         with open("data/expiry_details.json", 'r') as file:
#             try:
#                 data = json.load(file)
#             except json.JSONDecodeError:
#                 data = []  # If file is empty or has invalid JSON
#     else:
#         data = []

#     # Create a new entry
#     new_entry = {
#         'expiry': expiry,
#         'mfg': mfg,
#         'batch_no': batch_no,
#         'object_name': object_name
#     }

#     # Append new entry
#     data.append(new_entry)

#     # Write back to the file
#     with open("data/expiry_details.json", 'w') as file:
#         json.dump(data, file, indent=4)

def append_to_json_file(expiry, mfg, batch_no, object_name):
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

    # Check for duplicates based on 'object_name'
    for index, entry in enumerate(data):
        if entry['object_name'] == object_name:
            # Update existing entry only if the new value is not "missing"
            if expiry != "missing":
                entry['expiry'] = expiry
            if mfg != "missing":
                entry['mfg'] = mfg
            if batch_no != "missing":
                entry['batch_no'] = batch_no
            if object_name != "missing":
                entry['object_name'] = object_name
            break
    else:
        res = requests.get("http://localhost:8000/get-sensor-data")
        in_sensor = res.json()['in_sensor']
        if in_sensor :
            data.append(new_entry)
        else :
            print("product info attempted to update after the process")

    # Write back to the file
    with open("data/expiry_details.json", 'w') as file:
        json.dump(data, file, indent=4)


def process_image(roi_image,name):
    error_messages = []
    result_text = ""

    try:
        # Check if the ROI image is valid
        if roi_image is None or roi_image.size == 0:
            raise ValueError("Invalid ROI image provided.")

        # Create a temporary file to save the ROI image
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            # Write the ROI image to the temporary file
            cv2.imwrite(temp_file.name, roi_image)
            temp_file_path = temp_file.name  # Save the path for later use

        # Upload the image file
        image_file = genai.upload_file(path=temp_file_path, display_name="Product Image")
        print(f"Uploaded file '{image_file.display_name}' as: {image_file.uri}")

        # Prompt the model to find expiry date, manufacture date, and batch number
        prompt = (
            "This is an image of a product. Please identify the expiry date, "
            "manufacture date, and batch number of the product. If anything is missing, just say 'missing'"
            "i need reply with a end word in next line"
            "OUTPUT SHOULD STRICTLY be like "
            " start Expiry date: xx/xx/xxxx end"
            " start Manufacture date: xx/xx/xxxx end"
            " start Batch number: xxxxxxxx end"
        )

        # Generate content using the model
        response = model.generate_content([image_file, prompt])
        result_text = response.text

    except Exception as e:
        error_messages.append(f"An unexpected error occurred: {e}")
    finally:
        # Clean up: remove the temporary file
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    # Print any error messages or the result
    if error_messages:
        for error in error_messages:
            print(f"Error: {error}")
            return None
    else:
        print("Result:")
        print(result_text)
        print("parsed output from the AI")
        full_data=result_text
        expiry_date=full_data.split("Expiry date")[1].split("end")[0].strip(":").strip()
        manufacture_date=full_data.split("Manufacture date")[1].split("end")[0].strip(":").strip()
        batch=full_data.split("Batch number")[1].split("end")[0].strip(":").strip()
        print(f"MFG : {manufacture_date}  EXP : {expiry_date}  Batch : {batch}  name : {name}")
        append_to_json_file(expiry_date,manufacture_date,batch,name)
        return expiry_date,manufacture_date,batch

