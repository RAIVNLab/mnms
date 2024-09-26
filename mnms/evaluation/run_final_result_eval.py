import json
import os 
import argparse
from datasets import load_dataset
from .evaluator import MNMEvaluator, MNMFinalResultEvaluator

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-dir', default=None, type=str, help="the directory to import prediction execution results from.", required=True)
    parser.add_argument('--gt-dir', default="mnms/execution/gt_results", type=str, help="the directory to import gt execution results from.")
    parser.add_argument('--output-csv', default=None, type=str, help="the csv file to save the output to.")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    selected_ids = next(os.walk(args.gt_dir))[1]
    print(len(selected_ids))
    evaluator = MNMFinalResultEvaluator(
        gt_dir=args.gt_dir, 
        pred_dir=args.pred_dir,
        selected_eval_ids=selected_ids[:3]
    )
    avg_score, df = evaluator.evaluate()
    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
    else:
        exp_id = os.path.basename(args.pred_dir)
        output_csv = f"{exp_id}-exec-metrics.csv"
        print(output_csv)
        df.to_csv(output_csv, index=False)

if __name__ == '__main__':
    main()