from typing import Dict, List, Literal, Optional, Tuple
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import Levenshtein

def flatten_labels(
    gt_labels: List[str], pred_labels: List[str], categories: Optional[List[str]] = None
) -> Tuple[List[int], List[int], List[str]]:
    """
    Flatten the ground truth and predicted labels into a single list of int values.
    """
    union = list(set(gt_labels + pred_labels))
    if categories:

        def gt_feature(label):
            if label in gt_labels and label in categories:
                return categories.index(label) + 1
            return 0

        def pred_feature(label):
            if label in pred_labels and label in categories:
                return categories.index(label) + 1
            return 0

    else:

        def gt_feature(label):
            return 1 if label in gt_labels else 0

        def pred_feature(label):
            return 1 if label in pred_labels else 0

    gt_flat = [gt_feature(label) for label in union]
    pred_flat = [pred_feature(label) for label in union]
    return gt_flat, pred_flat, union

class EditDistMetrics:
    """
    Class to calculate normalized edit distance for the MNM task.
    """
    def __init__(
        self,
        categories: Optional[List[str]],
        normalized: bool = True,
    ):
        self.categories = categories
        self.cat2label = {cat: i + 1 for i, cat in enumerate(categories)}
        self.normalized = normalized
        self._gt: List[int] = []
        self._pred: List[int] = []

    def update(self, gt_labels: List[str], pred_labels: List[str]):
        # Turn tool names into numerical labels starting from 1, or 0 if a tool name does not exist
        self._gt.append([self.cat2label.get(name, 0) for name in gt_labels])
        self._pred.append([self.cat2label.get(name, 0) for name in pred_labels])

    def compute(self) -> Dict[str, float]:
        assert len(self._gt) == len(self._pred)
        total = 0
        for gt, pred in zip(self._gt, self._pred):
            total += Levenshtein.ratio(pred, gt)
        ned = 1 - total / len(self._gt)
        return {
            "normalized_edit_distance": round(ned * 100, 2)
        }


class PlanAccMetrics:
    """
    Class to calculate plan accuracy for the MNM task.
    """
    def __init__(
        self
    ):
        self._gt: List[int] = []
        self._pred: List[int] = []

    def update(self, gt_labels: List[str], pred_labels: List[str]):
        self._gt.append(gt_labels)
        self._pred.append(pred_labels)

    def compute(self) -> Dict[str, float]:
        # Binarize gt and pred, treating each gt plan as 1
        binary_gt = [1.0] * len(self._gt)
        binary_pred = []
        for gt, pred in zip(self._gt, self._pred):
            binary_pred += ([1.0] if pred == gt else [0.0])
        acc = accuracy_score(binary_gt, binary_pred)
        return {
            "accuracy": round(100 * acc, 2)
        }


class PRFMetrics:
    """
    Class to calculate the precision, recall, f1 metrics for the MNM task.
    """

    def __init__(
        self,
        categories: Optional[List[str]],
        average: Optional[Literal["micro", "macro", "binary"]],
    ):
        self.categories = categories
        self.average = average
        self.labels = list(range(1, len(categories) + 1)) if categories else None
        self._gt_flat: List[int] = []
        self._pred_flat: List[int] = []
        self._union: List[str] = []

    def update(self, gt_labels: List[str], pred_labels: List[str]):
        gt_flat, pred_flat, union = flatten_labels(
            gt_labels, pred_labels, categories=self.categories
        )
        self._gt_flat.extend(gt_flat)
        self._pred_flat.extend(pred_flat)
        self._union.extend(union)

    def compute(self) -> Dict[str, float]:
        precision, recall, f1, _ = precision_recall_fscore_support(
            self._gt_flat, self._pred_flat, labels=self.labels, average=self.average
        )
        return {
            "precision": round(100 * precision, 2),
            "recall": round(100 * recall, 2),
            "f1": round(100 * f1, 2),
        }