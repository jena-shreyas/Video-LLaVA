import json
import os
import sys
import os.path as osp
from typing import List, Dict
from tqdm import tqdm

def create_prompts(data: Dict):
    qn = data['question']
    qn_prompt = f"Question: {qn}\n"
    options = [
        f"{c}. " + data[f'a{i}'] for i, c in enumerate(['a', 'b', 'c', 'd', 'e'])
    ]   

    video_token = '<video>'
    answer_prompt = f"Answer: \n{video_token}"
    task_prompt = f"Answer the question with the option's letter from the given choices directly.\n\n" # Since the question is based on the video, try to use the video context as much as possible when checking the options.
    human_prompt = task_prompt + qn_prompt + "Options:\n\n" + "\n".join(options) + "\n\n" + answer_prompt
    ans = int(data['answer'])
    idx2letter_map = {
        0: "Option A",
        1: "Option B",
        2: "Option C",
        3: "Option D",
        4: "Option E"
    }
    gpt_prompt = idx2letter_map[ans]
    return human_prompt, gpt_prompt

def convert_data_format(data: List | Dict,
                        video_dir: str) -> List:
    new_data = []
    if isinstance(data, Dict):
        data_ = data.items()
    else:
        data_ = data
    for idx, d in tqdm(enumerate(data_)):
        if isinstance(data, Dict):
            _, data_dict = d
        else:
            data_dict = d
        data_dict['id'] = str(idx)
        data_dict['video'] = data_dict['video'] + '.mp4'
       
        human_prompt, gpt_prompt = create_prompts(data_dict)
        data_dict['conversations'] = [
            {'from': 'human', 'value': human_prompt},
            {'from': 'gpt', 'value': gpt_prompt}
        ]
        new_data.append(data_dict)

    return new_data


if __name__ == "__main__":
    filename = sys.argv[1]
    print(filename)
    video_dir = 'causalvidqa/videos'

    with open(f'causalvidqa/annotations/{filename}_old.json', 'r') as f:
        orig_data = json.load(f)

    conv_data = convert_data_format(orig_data, video_dir)

    with open(f'causalvidqa/annotations/{filename}_new.json', 'w') as f:
        json.dump(conv_data, f)
