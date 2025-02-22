import json

# Define the file path and the output path
input_file_path = "D:/code/MedicalQA/1_translate/cn_train_15W_20230918.json"
output_file_path = "D:/code/MedicalQA/二次检索任务/1 原始数据.json"

# Load the JSON file and save the first 3000 entries
try:
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        data = json.load(input_file)
    
    # Extract the first 3000 entries
    subset_data = data[:3000]
    
    # Save the subset to a new file
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump(subset_data, output_file, ensure_ascii=False, indent=4)
        
    result_message = f"The subset of the first 3000 entries has been saved to {output_file_path}."
except Exception as e:
    result_message = f"An error occurred: {str(e)}"

result_message
