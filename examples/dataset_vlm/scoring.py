"""Scoring a model's answers against a dataset's ground truth.

Two scorers, one per task of the dataset source:

- VQA. :func:`normalize_answer` is a port of the answer normalisation in the
  official VQA v2.0 evaluation code (``PythonEvaluationTools/vqaEvaluation/
  vqaEval.py`` at https://github.com/GT-Vision-Lab/VQA): lower-case, strip
  periods, canonicalise numbers and articles, expand a fixed contraction table,
  and drop punctuation. :func:`vqa_accuracy` is the official VQA accuracy of one
  prediction against the human answer list: ``min(matches / 3, 1)`` averaged over
  the leave-one-out subsets, which reduces to ``min(count_of_matches / 3, 1)``.
  With a short answer list (fewer than ten) the same formula is applied directly;
  it is the metric's own definition, stated for a smaller sample.

- Detection. :func:`match_detections` greedily matches predicted boxes to
  ground-truth boxes by IoU at a single threshold (COCO's AP is averaged over
  several thresholds; a single-threshold match is the simplest documented
  variant and is what this example reports). It returns precision, recall and F1
  for one image, from which a run's mean is taken. Boxes are ``[x, y, w, h]``.
"""

import re
from typing import Dict, List, Sequence

# --- VQA answer normalisation (ported from the official vqaEval.py) ----------

_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
    "hadnt": "hadn't", "hasnt": "hasn't", "havent": "haven't", "hes": "he's",
    "im": "i'm", "isnt": "isn't", "its": "it's", "lets": "let's", "shes": "she's",
    "shouldnt": "shouldn't", "thats": "that's", "theres": "there's", "theyre": "they're",
    "wasnt": "wasn't", "werent": "weren't", "whats": "what's", "wheres": "where's",
    "whos": "who's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't",
    "youll": "you'll", "youre": "you're", "youve": "you've",
}
_NUMBER_MAP = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
_ARTICLES = {"a", "an", "the"}
_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(\,)(\d)")
_PUNCT = [
    ";", "/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-",
    ">", "<", "@", "`", ",", "?", "!",
]


def _process_punctuation(text: str) -> str:
    out = text
    for punct in _PUNCT:
        if (punct + " " in text or " " + punct in text) or (_COMMA_STRIP.search(text) is not None):
            out = out.replace(punct, "")
        else:
            out = out.replace(punct, " ")
    out = _PERIOD_STRIP.sub("", out, re.UNICODE)
    return out


def _process_digit_article(text: str) -> str:
    words = []
    for word in text.lower().split():
        word = _NUMBER_MAP.get(word, word)
        if word in _ARTICLES:
            continue
        words.append(_CONTRACTIONS.get(word, word))
    return " ".join(words)


def normalize_answer(answer: str) -> str:
    """Return the VQA-normalised form of a raw answer string."""
    text = (answer or "").replace("\n", " ").replace("\t", " ").strip().lower()
    text = _process_punctuation(text)
    text = _process_digit_article(text)
    return text


def vqa_accuracy(prediction: str, answers: Sequence[str]) -> float:
    """Return the VQA accuracy of ``prediction`` against the human ``answers``.

    ``min(number of human answers matching the prediction / 3, 1)``: a prediction
    that three or more annotators gave scores 1.0, two scores 2/3, one 1/3.
    """
    pred = normalize_answer(prediction)
    gts = [normalize_answer(a) for a in answers]
    if not gts:
        return 0.0
    matches = sum(1 for gt in gts if gt == pred)
    return min(matches / 3.0, 1.0)


# --- detection matching ------------------------------------------------------

def iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Return the IoU of two ``[x, y, w, h]`` boxes."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2, ay2, bx2, by2 = ax + aw, ay + ah, bx + bw, by + bh
    inter_x = max(0.0, min(ax2, bx2) - max(ax, bx))
    inter_y = max(0.0, min(ay2, by2) - max(ay, by))
    inter = inter_x * inter_y
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def match_detections(
    predictions: List[Dict], ground_truth: List[Dict], iou_thresh: float = 0.5
) -> Dict[str, float]:
    """Greedily match predicted boxes to ground-truth boxes by IoU.

    ``predictions`` and ``ground_truth`` are lists of ``{"bbox": [x, y, w, h],
    "label": str}``. A prediction matches a ground-truth box of the same label
    with IoU at or above ``iou_thresh``; each ground-truth box matches at most
    once, highest IoU first. Returns ``tp``, ``fp``, ``fn`` and the per-image
    ``precision``, ``recall`` and ``f1``.
    """
    pairs = []
    for pi, pred in enumerate(predictions):
        for gi, gt in enumerate(ground_truth):
            if pred.get("label") and gt.get("label") and pred["label"] != gt["label"]:
                continue
            score = iou(pred["bbox"], gt["bbox"])
            if score >= iou_thresh:
                pairs.append((score, pi, gi))
    pairs.sort(reverse=True)
    used_pred, used_gt = set(), set()
    tp = 0
    for _score, pi, gi in pairs:
        if pi in used_pred or gi in used_gt:
            continue
        used_pred.add(pi)
        used_gt.add(gi)
        tp += 1
    fp = len(predictions) - tp
    fn = len(ground_truth) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
