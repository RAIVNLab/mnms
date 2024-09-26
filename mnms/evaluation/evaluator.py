import re
import os
import ast
import torch
import sys
sys.path.append('/gscratch/krishna/zixianma/miniconda3/envs/sg/lib/python3.9/site-packages')
sys.path.append('/gscratch/krishna/zixianma/miniconda3/lib/python3.10/site-packages')
# sys.path.append('./')
import json
import pickle
import tempfile
import traceback
import open_clip
import pandas as pd
from PIL import Image
from typing import Any, Dict, List, Literal, Tuple, Optional
from functools import partial
from .metrics import PlanAccMetrics, PRFMetrics, EditDistMetrics
from mnms.constants import TOOL_METADATA
from nltk.translate.bleu_score import sentence_bleu
from skimage.metrics import mean_squared_error as mse
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from datasets import load_dataset
from tqdm import tqdm


def find_best_match(gt_code_list: List[str], pred_code: str):
    """
    Find the best match between a list of ground truth codes and predicted code.
    """
    labeler = partial(get_code_labels, include_arg_names=False)
    gt_labels_list = [labeler(gt_code) for gt_code in gt_code_list]
    pred_labels = labeler(pred_code)
    match_scores = []
    for gt_labels in gt_labels_list:
        metric = PRFMetrics(categories=None, average="binary")
        metric.update(gt_labels, pred_labels)
        match_scores.append(metric.compute()["f1"])

    best_match_idx = match_scores.index(max(match_scores))
    return gt_code_list[best_match_idx]

def replace_node_with_placeholder(text):
    """
    Example:
    input: 'prefix for node-1.bleh node-2.blah'
    output: ('prefix for {} {}', [(1, 'bleh'), (2, 'blah')])
    """
    # This regular expression matches '<node-k>.blah' patterns
    pattern = r"<node-\d+>\.\w+"
    matches = re.findall(pattern, text)
    format_args = []
    for match in matches:
        if match.startswith("<node-"):
            node_id = int(match[6 : match.index(">")])
            arg_name = match.split(".")[-1]
            format_args.append((node_id, arg_name))
    # Replace matched patterns with '{}'
    replaced_text = re.sub(pattern, "{}", text)
    return replaced_text, format_args


def nodes_to_code(nodes):
    """
    Turns the json-format nodes into python code lines (without the solve() header and output used in code_str)
    Example:
    Input:
    [{"id": 0, "name": "image captioning", "args": {"image": "2327921.jpg"}}, 
    {"id": 1, "name": "text summarization", "args": {"text": output0["text"]}}]
    Output:
    output0 = image_captioning(image="2327921.jpg")
    output1 = text_summarization(text=output0['text'])
    """
    assert isinstance(nodes, list), f"nodes is of type {type(nodes)} but expected list"

    code = []
    for node in nodes:
        node_name = "output{}".format(node["id"])

        fn_name = node["name"].replace(" ", "_")
        arg_strs = []
        for arg_name, arg_value in node["args"].items():
            if isinstance(arg_value, str):
                if arg_value.startswith("<node-") and " " not in arg_value:
                    node_id = int(arg_value[6 : arg_value.index(">")])
                    node_key = arg_value[arg_value.index(".") + 1 :]
                    arg_value = f"output{node_id}['{node_key}']"
                elif "<node-" in arg_value:
                    placeholder_str, node_infos = replace_node_with_placeholder(
                        arg_value
                    )
                    node_values = []
                    for node_id, node_key in node_infos:
                        node_values.append(f"output{node_id}['{node_key}']")
                    arg_value = placeholder_str.format(*node_values)
                    arg_value = arg_value.replace('"', '\\"')
                    arg_value = 'f"{}"'.format(arg_value)
                else:
                    arg_value = arg_value.replace('"', '\\"')
                    arg_value = f'"{arg_value}"'
            else:
                # Turn argument values for e.g. number, year etc into strings
                arg_value = f'"{arg_value}"'

            arg_strs.append(f"{arg_name}={arg_value}")
        arg_str = ", ".join(arg_strs)
        code.append(f"{node_name} = {fn_name}({arg_str})")
    return "\n".join(code)


def extract_func_info_from_single_line(code):
    """
    Example:
    input: output1 = object_detection(image=output0['image'])
    output: {
        'output_var': 'output1',
        'func_name': 'object_detection',
        'args': {'image': 'output0[\'image\']'}}
    """
    # Parse the given code to an AST
    try:
        tree = ast.parse(code)

        # Initialize the result dictionary
        result = {"output_var": "", "func_name": "", "args": {}}

        # Find the first assignment in the code
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                # Extract the node name (variable name on the left of the assignment)
                result["output_var"] = node.targets[0].id

                # Check if the value is a function call
                if isinstance(node.value, ast.Call):
                    if isinstance(node.value.func, ast.Name):
                        result["func_name"] = node.value.func.id

                    # Extract the function arguments
                    for arg in node.value.args:
                        # This example doesn't handle positional arguments, but could be extended
                        pass
                    for keyword in node.value.keywords:
                        result["args"][keyword.arg] = ast.unparse(keyword.value)

                break  # Assuming only one assignment of interest per line of code

        if result["output_var"] == "" or result["func_name"] == "":
            return None
    except Exception as e:
        # print(e)
        # print(code)
        return None
    return result


def extract_func_info_from_code(code):
    info = []
    for line in code.split("\n"):
        result = extract_func_info_from_single_line(line.strip())
        if result:
            info.append(result)

    return info

def func_info_to_code(info):
    code = []
    for node in info:
        node_name = node["output_var"]
        fn_name = node["func_name"]
        arg_strs = []
        for arg_name, arg_value in node["args"].items():
            arg_value = f"{arg_value}"
            arg_strs.append(f"{arg_name}={arg_value}")
        arg_str = ", ".join(arg_strs)
        code.append(f"{node_name} = {fn_name}({arg_str})")
    return "\n".join(code)

def get_code_labels(
    code: str, include_arg_names=False, include_arg_values=False, edge_as_label=False, sep="--"
):
    """
    Example input:
    output0 = image_captioning(image="2327921.jpg")
    output1 = text_summarization(text=output0['text'])
    output2 = question_answering(text=output1['text'], question="What is the main dessert mentioned in the caption?")
    """

    parse = extract_func_info_from_code(code)
    if not include_arg_names and include_arg_values is True:
        raise ValueError("Cannot include arg values without arg names")

    labels = []
    for i, func_info in enumerate(parse):
        if "func_name" not in func_info:
            continue
        if edge_as_label:
            if i + 1 >= len(parse) or "func_name" not in parse[i+1]:
                continue
            label = [parse[i]["func_name"], parse[i+1]['func_name']]
        else:
            label = [func_info["func_name"]]

            if include_arg_names:
                # Sort the arguments by names to align the pred's and label's orderings 
                sorted_args = {val[0] : val[1] for val in sorted(func_info["args"].items(), key = lambda x: x[0])}
                for arg_name, arg_value in sorted_args.items():
                    if include_arg_values:
                        if isinstance(arg_value, str):
                            arg_value = arg_value.lower()    
                        label.append(f"{arg_name}={arg_value}")
                    else:
                        label.append(arg_name)

        labels.append(sep.join(label))

    return labels

def list_directories(path) -> List[str]:
    """List all directories within a given path."""
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]


def image_similarity(gt_image_path, pred_image_path, method="clip"):
    gt_image = Image.open(gt_image_path)
    pred_image = Image.open(pred_image_path)
    if method == "mse":
        score = mse(np.asarray(gt_image), np.asarray(pred_image))
    elif method == "clip":
        score = get_similarity_between_images(gt_image, pred_image)
    return score

def get_similarity_between_images(gt_image, pred_image):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14-378-quickgelu', pretrained='dfn5b', device=device)
    gt_image = preprocess(gt_image).unsqueeze(0)
    pred_image = preprocess(pred_image).unsqueeze(0)

    with torch.no_grad(), torch.cuda.amp.autocast():
        gt_image_features = clip_model.encode_image(gt_image.to(device))
        pred_image_features = clip_model.encode_image(pred_image.to(device))
        
        gt_image_features /= gt_image_features.norm(dim=-1, keepdim=True)
        pred_image_features /= pred_image_features.norm(dim=-1, keepdim=True)
    
    similarity = gt_image_features @ pred_image_features.T
    return similarity[0,0].item()

def bleu(reference, candidate):
    # Calculating BLEU score
    ref_tokens = [reference.lower().split()]
    cand_tokens = candidate.lower().split()
    n = len(cand_tokens)
    if n == 0:
        return 0
    if n < 4:
        weights = [1/n] * n
    else:
        weights = [0.25] * 4
    score = sentence_bleu(ref_tokens, cand_tokens, weights=weights)
    return score


def get_average_precision(gt_objects, pred_objects, temp_dir = "/mmfs1/gscratch/krishna/zixianma/mnms/mnms/execution/temp_data"):
    if len(pred_objects) == 0 or len(pred_objects[0]) == 0: # no predicted object
        return 0
    gt_objects = convert_objects_into_coco_format(gt_objects, gt=True)
    
    pred_objects = convert_objects_into_coco_format(pred_objects, gt=False)
    with tempfile.NamedTemporaryFile(mode='w+', dir=temp_dir, suffix=".json", delete=False) as temp_file:
        json.dump(gt_objects, temp_file)
        temp_filename = temp_file.name
        temp_file.seek(0)
        cocoGt = COCO(temp_filename)
    
    with tempfile.NamedTemporaryFile(mode='w+', dir=temp_dir, suffix=".json", delete=False) as temp_file:
        json.dump(pred_objects, temp_file)
        temp_filename = temp_file.name
        temp_file.seek(0)
        cocoDt = cocoGt.loadRes(temp_filename)
    score = get_average_precision_at_iou_50(cocoGt, cocoDt)
    return score   
    
def get_average_precision_at_iou_50(cocoGt, cocoDt):
    # Create COCO Eval object with the annotations and the detections
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    p = cocoEval.params
    # Get Average Precision (AP)
    # print('Average Precision (AP) for IoU threshold 0.5:', cocoEval.stats[1])
    return cocoEval.stats[1]

def convert_objects_into_coco_format(objects, gt=False):
    obj_list = []
    image_list = []
    image_id = 1
    categories = {}
    for i, object in enumerate(objects):
        id = i + 1
        obj_cat = object["label"]
        if obj_cat in categories:
            cat_id = categories[obj_cat]
        else:
            cat_id = (max(categories.values()) + 1) if len(categories) > 0 else 0
            categories[obj_cat] = cat_id 
        x, y = int(object["bbox"][0]), int(object["bbox"][1])
        width, height = int(object["bbox"][2] - object["bbox"][0]), int(object["bbox"][3] - object["bbox"][1])
        bbox = [x, y, width, height]
        area = width * height
        
        obj_dict = {
            "id": id,
            "image_id": image_id,
            "category_id": cat_id,
            "bbox": bbox,
            "area": int(area),
            "iscrowd": 0
        }
        if not gt:
            obj_dict["score"] = object['score'] if 'score' in object else 1.0

        img_dict = {"id": image_id}
        obj_list.append(obj_dict)
        image_list.append(img_dict)

    cat_list = []
    if gt:
        for obj_cat, cat_id in categories.items():
            cat_dict = {"id": cat_id, "name": obj_cat}
            cat_list.append(cat_dict)

        final_output = {"categories": cat_list, "images": image_list, "annotations": obj_list}
    else:
        final_output = obj_list
    return final_output 


class MNMEvaluator:
    def __init__(
        self,
        gt_data: List[Dict[str, Any]],
        pred_data: List[Dict[str, Any]],
        eval_set: Literal["gt", "pred"],
        plan_format: Literal["code", "json"],
        num_tools: int = 33,
        use_best_match: bool = False,
    ) -> None:
        self.gt_data = gt_data
        self.pred_data = pred_data
        self.eval_set = eval_set
        self.plan_format = plan_format
        self.num_tools = num_tools
        self.use_best_match = use_best_match
        self.categories = self.get_categories()
        assert (
            len(self.categories) == self.num_tools
        ), "Found {} categories, expected {}".format(
            len(self.categories), self.num_tools
        )
        self.metrics = {
            "tool_macro": PRFMetrics(categories=self.categories, average="macro"), 
            "argname": PRFMetrics(categories=None, average="binary"),
            "argvalue": PRFMetrics(categories=None, average="binary"),
            "edge": PRFMetrics(categories=None, average="binary"),
            "plan_tool": PlanAccMetrics(),
            "plan_argname": PlanAccMetrics(),
            "plan_argvalue": PlanAccMetrics(),
            "tool": EditDistMetrics(categories=self.categories),
        }
        self.code_labelers = {
            "tool_macro": partial(get_code_labels),
            "argname": partial(get_code_labels, include_arg_names=True),
            "argvalue": partial(get_code_labels, include_arg_names=True, include_arg_values=True),
            "edge": partial(get_code_labels, edge_as_label=True),
            "plan_tool": partial(get_code_labels),
            "plan_argname": partial(get_code_labels, include_arg_names=True),
            "plan_argvalue": partial(get_code_labels, include_arg_names=True, include_arg_values=True),
            "tool": partial(get_code_labels),
        }
        self.wrong_predictions = {"plan_tool": [], "plan_argname": [], "plan_argvalue": []}

    def get_categories(self):
        categories = []
        for tool in TOOL_METADATA.keys():
            categories.append(tool.replace(" ", "_"))
        return categories

    def get_paired_eval_data(self):
        if self.eval_set == "gt":
            eval_ids = [sample["id"] for sample in self.gt_data]
        elif self.eval_set == "pred":
            eval_ids = [sample["id"] for sample in self.pred_data]
        else:
            raise ValueError("Invalid eval_set")

        id2gt = {
            sample["id"]: sample for sample in self.gt_data if sample["id"] in eval_ids
        }
        id2pred = {
            sample["id"]: sample
            for sample in self.pred_data
            if sample["id"] in eval_ids
        }
        for sample_id in eval_ids:
            yield id2gt[sample_id], id2pred[sample_id]

    def evaluate(self) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        invalid_predictions = []
        for gt_sample, pred_sample in self.get_paired_eval_data():
            try:
                pred_code = pred_sample["prediction"] if self.plan_format == 'code' else nodes_to_code(pred_sample["prediction"])
            except Exception as e:
                print(pred_sample["prediction"])
                print(e, '\n')
                pred_code = ""
                invalid_predictions.append(
                    dict(gt=gt_sample, pred=pred_sample, err=traceback.format_exc())
                )
            if not isinstance(pred_code, str):
                pred_code = str(pred_code)
            if self.use_best_match:
                plans = [eval(gt_sample["plan_str"])] + eval(gt_sample["alt_plans_str"])
                gt_code_list = [nodes_to_code(plan) for plan in plans]
                gt_code = find_best_match(gt_code_list, pred_code)
            else:
                gt_code = nodes_to_code(eval(gt_sample["plan_str"]))
            
            for metric_name, metric in self.metrics.items():
                gt_labels = self.code_labelers[metric_name](gt_code)
                pred_labels = self.code_labelers[metric_name](pred_code)
                metric.update(gt_labels, pred_labels)
                # Record wrong predictions
                if metric_name in ["plan_tool", "plan_argname", "plan_argvalue"] and pred_labels != gt_labels:
                    self.wrong_predictions[metric_name].append({'id': gt_sample['id'], 'pred': pred_labels, 'label': gt_labels})
                

        records = []
        for metric_name, metric in self.metrics.items():
            metrics: Dict[str, Any] = metric.compute()
            for k, v in metrics.items():
                record = {}
                record["metric"] = f"{metric_name}_{k}"
                record["value"] = v

                records.append(record)

        df = pd.DataFrame.from_records(
            records, columns=["metric", "value"]
        )
        return df, invalid_predictions


class MNMFinalResultEvaluator:
    def __init__(
        self,
        gt_dir: str,
        pred_dir: str,
        selected_eval_ids: Optional[List] = None
    ) -> None:
        self.gt_dir = gt_dir
        self.pred_dir = pred_dir
        self.all_out_keys = set()
        for tool, tool_meta in TOOL_METADATA.items():
            out_keys = tool_meta['output']['arg_name']
            self.all_out_keys.update(out_keys)
        self.selected_eval_ids = selected_eval_ids 

    def evaluate(self) -> Tuple[float, pd.DataFrame]:
        gt_results, _ = self.extract_all_results(self.gt_dir)
        pred_results, _ = self.extract_all_results(self.pred_dir)
        all_task_ids = self.selected_eval_ids if self.selected_eval_ids else list(gt_results.keys())

        scores = []
        for task_id in tqdm(all_task_ids):
            task_id = str(task_id)
            if task_id not in gt_results:
                print(f"skipping {task_id} as it's not in gt execution results")
                continue
            if task_id not in pred_results:
                # score is defined as 0 if no execution result is found in prediction directory
                final_score = 0
            else:
                gt_res = gt_results[task_id]
                pred_res = pred_results[task_id]
                
                results = {"gt": gt_res, "pred": pred_res}
                final_results = {}
                for key, res in results.items():
                    final_ans = {}
                    for node, node_res in res.items():
                        for res_key, res_content in node_res.items():
                            if res_key in final_ans:
                                final_ans[res_key].append(res_content)
                            else:
                                final_ans[res_key] = [res_content]
                    final_results[key] = final_ans
                final_score = self.score_final_results(task_id, final_results['gt'], final_results['pred'])
            # print("FINAL score:", final_score)
            scores.append({'task_id': task_id, 'avg_score': round(final_score, 4)})

        df = pd.DataFrame.from_records(scores)
        avg_score = df['avg_score'].mean()
        return avg_score, df
    
    def score_final_results(self, task_id, gt_result, pred_result) -> float:
        all_out_keys = self.all_out_keys
        all_scores = []
        for res_key, gt_content in gt_result.items():
            if res_key not in pred_result or res_key not in all_out_keys:
                print(f"skipping {res_key}, because it's not in {all_out_keys} or {pred_result.keys()}")
                continue
            pred_content = pred_result[res_key]
            if res_key == "text" or (isinstance(gt_content, str) and isinstance(pred_content, str)):
                gt_content = '\n'.join(gt_content)
                pred_content = '\n'.join(pred_content)
            
            content_score = self.eval_content_by_key(task_id, res_key, gt_content, pred_content)
            if content_score:
                all_scores.append(content_score)
        total_score = np.mean(all_scores) if len(all_scores) > 0 else 0
        return total_score

    def extract_all_results(self, result_dir) -> Tuple[Dict[str, Dict], List[str]]:
        all_task_ids = list_directories(result_dir)
        all_results = {}
        invalid_ids = set()
        for task_id in all_task_ids:
            results = {}
            results_path = os.path.join(result_dir, task_id, "result.json")
            if os.path.exists(results_path):
                node2path = json.load(open(results_path, "r"))
                for node_id, node_path in node2path.items():
                    if not os.path.exists(node_path): # the tool node's output path might be a relative path
                        node_path = os.path.join(result_dir, task_id, node_path)
                        if not os.path.exists(node_path):
                            invalid_ids.add(task_id)
                            continue
                    if node_path.endswith(".json"):
                        node_results = json.load(open(node_path, "r"))
                    elif node_path.endswith(".pkl"):
                        node_results = pickle.load(open(node_path, "rb"))
                    else:
                        raise NotImplementedError
                    results[node_id] = node_results
            all_results[task_id] = results
        return all_results, invalid_ids
    

    def eval_content_by_key(self, task_id, key, gt_content, pred_content) -> float:
        if key == "text" or (isinstance(gt_content, str) and isinstance(pred_content, str)):
            score = bleu(gt_content, pred_content)
            print(f"BELU score: {score}")
            return score
        elif key == "image":
            scores = []
            for gt_img, pred_img in zip(gt_content, pred_content):
                gt_img = os.path.join(self.gt_dir, task_id, gt_img) # because the gt_img is a relative path
                if (gt_img and pred_img) and os.path.exists(gt_img) and os.path.exists(pred_img):
                    score = image_similarity(gt_img, pred_img)
                else:
                    score = 0
                scores.append(score)
            final_score = np.mean(scores)
            return final_score
        elif key in ["objects", "object"]:
            gt_objects = gt_content[0]
            pred_objects = pred_content[0]
            if len(gt_objects) == 0:
                return None
            exact_match_score = int(str(gt_objects) == str(pred_objects))
            if key == "object":
                gt_objects = [gt_objects]
                pred_objects = [pred_objects]

            if 'label' in gt_objects[0] and 'bbox' in gt_objects[0]:
                ap_score = get_average_precision(gt_objects, pred_objects)
            else:
                ap_score = 0
            score = max(exact_match_score, ap_score)
            return score
        else:
            # if str(gt_content) != str(pred_content):
            #     print(f"results for {key}:")
            #     print("GT:", gt_content)
            #     print("PRED:", pred_content)
            res = int(str(gt_content) == str(pred_content))
        return res


        
