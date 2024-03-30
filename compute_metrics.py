import os
import os.path as osp
import json
import re
from tqdm import tqdm
from typing import List, Dict
from pprint import pprint

if __name__ == "__main__":
    checkpoint = "videollava-7b"
    type = "annotated"
    pred_dir = f"results/eval/{checkpoint}/causalvidqa/predictions/{type}"
    with open(osp.join(pred_dir, "1_0.json")) as f:
        preds = json.load(f)

    data_dir = "data/causalvidqa/annotations"
    with open(osp.join(data_dir, f"data_{type}.json"), 'r') as f:
        data_ : List = json.load(f)

    data = {}
    for sample in data_:
        data[sample["qid"]] = sample

    cat_acc = {
        cat: {
            "correct": 0,
            "total": 0,
            "acc": 0.0
        }
        for cat in ["descriptive", "explanatory", "predictive", "counterfactual"]
    }

    for sample in tqdm(preds, total=len(preds)):
        question = sample["question"]
        qid = sample["id"]
        qtype = qid.split("_")[-1]
        answer = sample["answer"]
        cat_acc[qtype]["total"] += 1
        opt2amap = {
            'a': 'a0',
            'b': 'a1',
            'c': 'a2',
            'd': 'a3',
            'e': 'a4'
        }
        pred = data[qid][opt2amap[sample["pred"].lower()]]
        if pred == answer:
            cat_acc[qtype]["correct"] += 1

    for cat in cat_acc:
        cat_acc[cat]["acc"] = cat_acc[cat]["correct"] / cat_acc[cat]["total"]

    pprint(cat_acc)


