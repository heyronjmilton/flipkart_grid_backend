import os
import time
import json
import cv2
from utils.gemini_image import process_image


def delete_all_files_in_folder(folder_path):
    try:
        # Check if the folder exists
        if os.path.exists("last_modified.json") :
            os.remove("last_modified.json")
        if not os.path.exists(folder_path):
            print(f"The folder '{folder_path}' does not exist.")
            return
        
        # Iterate through the files in the folder
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            
            # Check if it is a file (not a folder)
            if os.path.isfile(file_path):
                os.remove(file_path)  # Delete the file
                print(f"Deleted: {file_path}")
        
        print("All files in the folder have been deleted.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

# Function to get modified files and their last modified times
def get_modified_files(folder_path):
    modified_files = {}
    # Iterate over all files in the folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            # Get the last modified time of the file
            modified_time = os.path.getmtime(file_path)
            modified_files[file_path] = modified_time
    return modified_files

# Function to save the last modified times
def save_last_modified_times(file_path, last_modified):
    with open(file_path, 'w') as f:
        json.dump(last_modified, f)

def display_image(image_path):
    try:
        img = cv2.imread(image_path)
        
        _ , name = image_path.split("\\")
        obj_name, _ = name.split(".")
        process_image(img,obj_name)
        print(f"Displayed image: {image_path}")
        
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")

# Function to load the last modified times
def load_last_modified_times(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)  # Return the loaded dictionary
    return {}  # Default to empty dictionary if file does not exist

def main():
    folder_path = 'details'  # Change this to your target folder
    delete_all_files_in_folder(folder_path)
    last_modified_file = 'last_modified.json'

    # Load the last modified times
    last_modified = load_last_modified_times(last_modified_file)

    while True:
        # Get the current modified files and their times
        current_modified_files = get_modified_files(folder_path)

        # Check for modified files
        for file_path, modified_time in current_modified_files.items():
            if file_path not in last_modified:
                # If the file is new, report it
                print(f"New file detected: {file_path}")
                display_image(file_path)
            elif modified_time > last_modified[file_path]:
                # If the file has been modified, report it
                print(f"Modified file: {file_path}")
                display_image(file_path)

        # Update last modified dictionary with current modified times
        last_modified.update(current_modified_files)

        # Save the last modified times
        save_last_modified_times(last_modified_file, last_modified)

        # Wait for a specified interval (e.g., 10 seconds) before checking again
        time.sleep(1)

if __name__ == "__main__":
    main()
