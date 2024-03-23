import re
import json
import shutil
import pickle
from PIL import Image
from .tool_api import *
from .config import *
from constants import *
from copy import deepcopy

def find_node_patterns(text):
    # The regular expression pattern to find the <node-x>.xxxx in the args
    pattern = r"<node-\d+>\.[a-zA-Z]+"
    # Find all occurrences of the pattern in the text
    matches = re.findall(pattern, text)
    return matches

def save_output(task_idx, node_idx, output_dict, result_folder):
    new_dict = deepcopy(output_dict)
    full_result_path = os.path.join(result_folder, str(task_idx))
    if not os.path.exists(full_result_path):
        os.makedirs(full_result_path)
    for key, output in output_dict.items():
        # Determine the type of output and set the appropriate file extension
        if isinstance(output, Image.Image):
            file_path = os.path.join(full_result_path, f"node-{node_idx}.jpg")
            output.save(file_path)
            new_dict[key] = file_path

    print("Output dict to be saved:", new_dict)
    output_dict_path_json = os.path.join(full_result_path, f"node-{node_idx}.json")
    output_dict_path_pickle = os.path.join(full_result_path, f"node-{node_idx}.pkl")
    try:
        # Try saving the output dictionary in a json file
        with open(output_dict_path_json, "w") as file:
            json.dump(new_dict, file)
        return output_dict_path_json
    except TypeError as e:
        # If saving into json failes, remove the json file and save it as a pickle file
        if os.path.exists(output_dict_path_json):
            os.remove(output_dict_path_json)
        with open(output_dict_path_pickle, "wb") as file:
            pickle.dump(new_dict, file)
        return output_dict_path_pickle


def save_result(idx, output, result_folder):
    # Create the result directory if it doesn't exist
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # Create the subfolder inside the result folder
    subfolder_path = os.path.join(result_folder, str(idx))
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)

    # Move each file from the output to the subfolder
    for file_path in output.values():
        if os.path.exists(file_path):
            # Extract the filename from the path
            file_name = os.path.basename(file_path)
            # Define the new path for the file
            new_path = os.path.join(subfolder_path, file_name)
            # Move the file
            shutil.move(file_path, new_path)
        else:
            print(f"File not found: {file_path}")
    
    output_file_path = os.path.join(subfolder_path, 'result.json')
    with open(output_file_path, 'w') as outfile:
        json.dump(output, outfile, indent=4)

def execute_node(node):
    if node['name'] in TOOL_METADATA:
        func_name = node['name'].replace(' ', '_')
        # Call the corresponding tool function and store the value in result
        func = globals()[func_name]
        result = func(**node['args'])
    else:
        raise ValueError(f"Unknown tool: {node['name']}")
    return result

def pipeline(task_id, task_nodes, result_folder=RESULT_PATH):
    '''
    Execute a list of tool nodes, and save the tool outputs.
    '''
    outputs = {}
    outputs_path = {}
    full_result_path = os.path.join(result_folder, str(task_id))
    for node in task_nodes:
        # Update task arguments with outputs from previous tasks
        processed_node = deepcopy(node)
        for arg_key, arg_value in node['args'].items():
            if isinstance(arg_value, str) and arg_value.find('<node-') > -1:
                # This arg value comes from last node's output
                patterns = find_node_patterns(arg_value)
                for pattern in patterns:
                    # Extract node id from the string
                    node_id = pattern[6]  
                    start = pattern.find('.')
                    last_node_out_arg_name = pattern[start+1:]

                    last_node_out_path_json = os.path.join(full_result_path, f"node-{node_id}.json")
                    last_node_out_path_pickle = os.path.join(full_result_path, f"node-{node_id}.pkl")
                    
                    if os.path.exists(last_node_out_path_json): 
                        # Find saved intermediate output
                        print(f'finding last output at {last_node_out_path_json}')
                        last_node_out = json.load(open(last_node_out_path_json, "r"))
                        # Only replace <node-i>.text with last output's text and directly use last output otherwise
                        new_arg_value = processed_node['args'][arg_key].replace(pattern, last_node_out[last_node_out_arg_name]) if last_node_out_arg_name == "text" else  last_node_out[last_node_out_arg_name]
                        processed_node['args'][arg_key] = new_arg_value 
                    elif os.path.exists(last_node_out_path_pickle):
                        print(f'finding last output at {last_node_out_path_pickle}')
                        last_node_out = pickle.load(open(last_node_out_path_pickle, "rb"))
                        new_arg_value = processed_node['args'][arg_key].replace(pattern, last_node_out[last_node_out_arg_name]) if last_node_out_arg_name == "text" else  last_node_out[last_node_out_arg_name]
                        processed_node['args'][arg_key] = new_arg_value 
                    else:
                        raise ValueError(f"Output from node {node_id} not found for task {task_id}")
            elif isinstance(arg_value, str) and arg_value.endswith(('.jpg','.png', '.jpeg', '.flac')):
                # If the input isn't generated by the tool, but it's a file path, then we need to get the file path from the data folder
                processed_node['args'][arg_key] = get_full_path_data(arg_value)  
            print(processed_node['args'][arg_key])
        # Execute the task
        result = execute_node(processed_node)
        # Save the output to result folder and return the path
        output_path = save_output(task_id, node['id'], result, result_folder=result_folder) 
        outputs[str(node['id'])] = result
        outputs_path[str(node['id'])] = output_path

    return {"value": outputs, "path": outputs_path}