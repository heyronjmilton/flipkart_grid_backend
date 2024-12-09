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
    items_sheet.append(list(items_sheet_headers))
    for row in data:
        items_sheet.append(list(row.values()))

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