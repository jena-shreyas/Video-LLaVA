import os
import os.path as osp
import torch
from transformers import AutoConfig
from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

def parse_outputs(outputs: str
                )-> str:

    # try:
    #     find_str = "Answer:"
    #     parsed_outputs = outputs[outputs.find(find_str)+len(find_str) : ].replace("</s>", "")
    #     return parsed_outputs
    # except Exception as e:
    #     return ""
    return outputs.replace("</s>", "")
    

def main():
    disable_torch_init()
    video = 'data/causalvidqa/videos/9t_37F60-Y4_000007_000017.mp4'
    # inp = "Answer the question with the option's letter from the given choices directly.\n\nQuestion: What is [person_1] going to do?\nOptions:\n\na. [person_1] is going to continue skateboarding.\nb. [person_1] is going to continue skiing.\nc. [person_1] is going to spin around and raise the sword.\nd. [person_1] is trying to crawl.\ne. [person_1] is going to continue stretching.\n\nAnswer: \n"
    inp = "Describe what is happening in this video."    # . Please note that some of the objects in the video may be completely covered by a monochromatic segmentation mask, so take special care in figuring out the actual type of these objects properly.
    model_path = "LanguageBind/Video-LLaVA-7B"     # ./checkpoints/videollava-7b-lora

    # initialize model config from a given model config file
    # config_path = osp.join(model_path, 'config.json')
    # config = AutoConfig.from_pretrained(config_path)

    cache_dir = 'cache_dir'
    device = 'cuda'
    load_4bit, load_8bit = True, False
    model_name = get_model_name_from_path(model_path)
    # model_base = "/home/shreyasjena/BTP/models/Video-LLaVA/cache_dir/models--LanguageBind--Video-LLaVA-7B/snapshots/b1d6a63f98cc93153d3e9ff295bd6dee5ffafe4c"
    tokenizer, model, processor, _ = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device, cache_dir=cache_dir)
    video_processor = processor['video']
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles

    video_tensor = video_processor(video, return_tensors='pt')['pixel_values']
    if type(video_tensor) is list:
        tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
    else:
        tensor = video_tensor.to(model.device, dtype=torch.float16)

    # print(f"Roles : {roles[1]}: {inp}")
    inp = ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames) + '\n' + inp
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    # print(f"Prompt : \n\n{prompt}")
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=tensor,
            do_sample=True,
            temperature=1,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    print(f"Outputs: \n{outputs}")

    # outputs = "Answer: \nASSISTANT: c</s>"
    # final_outputs = parse_outputs(outputs)
    # print(f"Parsed outputs : \n{final_outputs}")

if __name__ == '__main__':
    main()