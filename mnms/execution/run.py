import os
import json
import argparse
from .executor import *
from .tool_api import *
from datasets import load_dataset

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', default=None, type=str, help="the file to import the input from.")
    parser.add_argument('--output-dir', default=None, type=str, help="the directory to save the output to.")
    parser.add_argument('--plan-format', default=None, type=str, choices=['json', 'code'], required=True)
    args = parser.parse_args()
    return args

def load_mnm_data_from_json(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            sample = json.loads(line)
            data.append(sample)
    return data

def extract_exp_id_from_json_filename(input_file):
    '''
    Find the unique experiment id. You should change this based on your custom file name
    '''
    start_idx = input_file.rfind('/') + 1
    end_idx = -len('.json')
    exp_id = input_file[start_idx:end_idx]
    return exp_id

def main():
    args = get_args()

    result_folder = args.output_dir if args.output_dir else "execution/results"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    
    if args.input_file:
        exp_id = extract_exp_id_from_json_filename(args.input_file)
        tasks = load_mnm_data_from_json(args.input_file)
    else:
        exp_id = "gt"
        tasks = load_dataset("zixianma/mnms", split="test_human_verified_filtered")
    # print(exp_id)
    # print(len(tasks))

    final_result_folder = os.path.join(result_folder, f"{exp_id}-{args.plan_format}")
    if args.plan_format == 'code':
        executor = CodeExecutor(result_folder=final_result_folder)
    else:
        executor = Executor(result_folder=final_result_folder)
    is_code_executor = isinstance(executor, CodeExecutor)
    
    output = os.path.join(result_folder, f"{exp_id}-{args.plan_format}-exec-results.json")
    has_executed = []
    # Record the ids of the examples that have been executed
    if os.path.exists(output):
        rf = open(output, "r")
        for line in rf:
            data = json.loads(line)
            has_executed.append(data["id"])
        rf.close()
    
    wf = open(output, "a")
    for task in tasks:
        idx = task['id']
        # Skip the examples that have been executed
        if idx in has_executed:
            continue
        print(f"Executing the plan for task {idx}...")
        if is_code_executor:
            plan = task['prediction'] if 'prediction' in task else task['code_str']
        else:
            plan = task['prediction'] if 'prediction' in task else eval(task['plan_str'])
        print(plan)
        result = executor.execute(idx, plan)

        if result['status']:
            result_dict = {'id': idx, 'res': 1, 'msg': 'success'}
        else:
            print(result['message'])
            result_dict = {'id': idx, 'res': 0, 'msg': result['message']}
            
        wf.write(json.dumps(result_dict) + "\n")
        wf.flush()


if __name__ == "__main__":
    main()