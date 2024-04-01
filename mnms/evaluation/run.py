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
    parser.add_argument('--eval-set', default='gt', type=str, choices=['gt', 'pred'])
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    
    gt_data = load_dataset("zixianma/mnms", split="test_human_verified_filtered")
    pred_data = load_mnm_data_from_json(args.preds_file)
    total_n = len(gt_data) if args.eval_set == "gt" else len(pred_data)

    evaluator = MNMEvaluator(
        gt_data=gt_data,
        pred_data=pred_data,
        eval_set=args.eval_set,
        plan_format=args.plan_format
    )
    df, invalid_predictions = evaluator.evaluate()
    print(f"There are {len(invalid_predictions)}/{total_n} invalid predictions that fail to be parsed correctly.")
    print(df)
    if args.output_csv:
        df.to_csv(args.output_csv, index=False)


if __name__ == '__main__':
    main()