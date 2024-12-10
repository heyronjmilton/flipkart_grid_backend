from fastapi import UploadFile
import os

UPLOAD_FOLDER = "VIDEO_UPLOADS"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

async def handle_file_upload(file: UploadFile, class_name: str, item_type: str):
    
    

    file_location = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_location, "wb") as f:
        content = await file.read()
        f.write(content)

    # Return the success message along with other form data
    return {
        "message": f"File '{file.filename}' uploaded successfully!",
        "class_name": class_name,
        "item_type": item_type,
    }