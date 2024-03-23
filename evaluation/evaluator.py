import re
import ast
import traceback
import pandas as pd
from typing import Any, Dict, List, Literal, Tuple
from functools import partial
from .metrics import PlanAccMetrics, PRFMetrics, EditDistMetrics
from constants import TOOL_METADATA

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
        # self.wrong_predictions = {"tool_micro": [], "tool_macro": [], "argname": [], "argvalue": []}

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
                # if pred_labels != gt_labels:
                #     self.wrong_predictions[metric_name].append({'pred': pred_labels, 'label': gt_labels})
                

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