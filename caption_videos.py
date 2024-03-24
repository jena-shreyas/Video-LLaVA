import os
import os.path as osp
import json
import torch
from tqdm import tqdm
from transformers import AutoConfig
from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
    
VIDEO_DIR = "data/causalvidqa/videos"
video2captions = {}

def main():
    disable_torch_init()
    prompt = "Describe what is happening in this video."
    model_path = "LanguageBind/Video-LLaVA-7B"     # ./checkpoints/videollava-7b-lora
    cache_dir = 'cache_dir'
    device = 'cuda'
    load_4bit, load_8bit = True, False
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, _ = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device, cache_dir=cache_dir)
    video_processor = processor['video']
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles

    prompt = ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames) + '\n' + prompt
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    for i, video_file in enumerate(tqdm(os.listdir(VIDEO_DIR), total=len(os.listdir(VIDEO_DIR)))):
        video = osp.join(VIDEO_DIR, video_file)
        video_tensor = video_processor(video, return_tensors='pt')['pixel_values']
        if type(video_tensor) is list:
            tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
        else:
            tensor = video_tensor.to(model.device, dtype=torch.float16)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=tensor,
                do_sample=True,
                temperature=0.1,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()

        video_name = video_file.split('.')[0]
        video2captions[video_name] = {
            'video': video,
            'caption': outputs
        }

        if i%10 == 0:
            with open('data/causalvidqa/video2captions.json', 'w') as f:
                json.dump(video2captions, f)

    with open('data/causalvidqa/video2captions.json', 'w') as f:
        json.dump(video2captions, f)


if __name__ == '__main__':
    main()