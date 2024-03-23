import json 
import argparse
from datasets import load_dataset
from .evaluator import MNMEvaluator

def load_mnm_data_from_json(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            sample = json.loads(line)
            data.append(sample)
    return data

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds-file', default=None, type=str, help="the json file to import the predictions from.", required=True)
    parser.add_argument('--output-csv', default=None, type=str, help="the csv file to save the output to.")
    parser.add_argument('--plan-format', default=None, type=str, choices=['json', 'code'], required=True)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    
    gt_data = load_dataset("zixianma/mnms", split="test_human_verified_filtered")
    print(gt_data)

    evaluator = MNMEvaluator(
        gt_data=gt_data,
        pred_data=load_mnm_data_from_json(args.preds_file),
        eval_set="gt",
        plan_format="json"
    )
    df, invalid_predictions = evaluator.evaluate()
    print(f"There are {len(invalid_predictions)} invalid predictions that fail to be parsed correctly.")
    print(df)
    if args.output_csv:
        df.to_csv(args.output_csv, index=False)


if __name__ == '__main__':
    main()