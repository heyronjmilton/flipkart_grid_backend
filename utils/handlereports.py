import os, openpyxl
from collections import Counter





def save_expiry_details_to_excel(data, folder, file_name):

    if not os.path.exists(folder):
        os.makedirs(folder)

    excel_file = os.path.join(folder, file_name)

    for row in data:
        row['object_name'] = row['object_name'].split("#")[0] #to remove the # values from the name of the object


    workbook = openpyxl.Workbook()

    items_sheet = workbook.active
    items_sheet.title = "Items Sheet"

    items_sheet_headers = list(data[0].keys())
    items_sheet_headers.insert(0, "SI no")  # Add a new header for serial numbers
    items_sheet.append(items_sheet_headers)

    for index, row in enumerate(data, start=1):  # Use enumerate to generate serial numbers starting from 1
        row_with_serial = [index] + list(row.values())  # Prepend the serial number to the row
        items_sheet.append(row_with_serial)

    for column in items_sheet.columns:
        max_length = 0
        column_letter = column[0].column_letter  # Get the column letter
        for cell in column:
            try:
                if cell.value:
                    max_length = max(max_length, len(str(cell.value)))
            except:
                pass
        adjusted_width = max_length + 2
        items_sheet.column_dimensions[column_letter].width = adjusted_width

    object_names = [row['object_name'] for row in data]
    object_name_counts = Counter(object_names)

    summary_sheet = workbook.create_sheet(title="Items Summary")
    summary_sheet.append(["Item Name","Count"])
    for object_name, count in object_name_counts.items():
        summary_sheet.append([object_name, count])
    
    for column in summary_sheet.columns:
        max_length = 0
        column_letter = column[0].column_letter  # Get the column letter
        for cell in column:
            try:
                if cell.value:
                    max_length = max(max_length, len(str(cell.value)))
            except:
                pass
        adjusted_width = max_length + 2
        summary_sheet.column_dimensions[column_letter].width = adjusted_width

    workbook.save(excel_file) 

def save_fruit_details_to_excel(data, folder, file_name):
    if not os.path.exists(folder):
        os.makedirs(folder)

    excel_file = os.path.join(folder, file_name)

    workbook = openpyxl.Workbook()

    fruits_sheet = workbook.active
    fruits_sheet.title = "Fruits and Vegetables Sheet"

    fruits_sheet_header = ["Item", "Quality", "Count"]
    fruits_sheet.append(fruits_sheet_header)
    for key,value in data.items():
        item_name = key.split("_")[1]
        item_quality = key.split("_")[0]
        item_count = value
        fruits_sheet.append([item_name, item_quality, item_count])
    
    for column in fruits_sheet.columns:
        max_length = 0
        column_letter = column[0].column_letter  # Get the column letter
        for cell in column:
            try:
                if cell.value:
                    max_length = max(max_length, len(str(cell.value)))
            except:
                pass
        adjusted_width = max_length + 2
        fruits_sheet.column_dimensions[column_letter].width = adjusted_width
    
    workbook.save(excel_file) 

    # print(f"fruit sheet header : {fruits_sheet_header}")
