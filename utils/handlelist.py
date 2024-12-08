import time, os, json

def make_object_final(object_name,file_name) : #expiry_details.json(packed products)

    if os.path.exists(f"data/{file_name}"):
        # Read the existing data
        with open(f"data/{file_name}", 'r') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = []  # If file is empty or has invalid JSON
    else:
        print(f"the details file is not yet created")


    for index, entry in enumerate(data):
        if entry['object_name'] == object_name:
            print("OBJECT DATA ")
            print(entry)
            new_object_name = object_name +'#'+str(time.time())
            print(f"NEW OBJECT NAME : {new_object_name}")
            entry['object_name'] = new_object_name





    with open(f"data/{file_name}", 'w') as file:
        json.dump(data, file, indent=4)