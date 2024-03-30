import json
import os
import os.path as osp
import sys
from typing import List

def convert_to_eval_format(data: List,
                        data_dir: str):
    qns = []
    answers = []
    for qn_dict in data:
        qns.append(
            {
                "video_name": qn_dict['video'].split('.')[0],
                "question_id": qn_dict['qid'],
                "question": qn_dict['conversations'][0]['value']
            }
        )

        cat2catids = {
            'descriptive': 0,
            'explanatory': 1,
            'predictive': 2,
            'counterfactual': 3,
        }

        answers.append({
            "answer": qn_dict['conversations'][1]['value'],
            "type": cat2catids[qn_dict['qn_type']],
            "question_id": qn_dict['qid']
        })

        with open(osp.join(data_dir, 'test_q.json'), 'w') as f:
            json.dump(qns, f, indent=4)

        with open(osp.join(data_dir, 'test_a.json'), 'w') as f:
            json.dump(answers, f, indent=4)

if __name__ == "__main__":
    input_file = sys.argv[1]    # data_annotated.json
    subfolder = input_file.split('.')[0].split('_')[1]
    data_dir = f"causalvidqa/annotations/{subfolder}"

    with open(osp.join(data_dir, input_file), "r") as f:
        data = json.load(f)

    convert_to_eval_format(data, data_dir)
    