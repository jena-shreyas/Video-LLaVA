

JSON_FOLDER="data/causalvidqa/annotations"
# IMAGE_FOLDER="llava_all_image_video"
VIDEO_FOLDER="data/causalvidqa/videos"

# HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 
 
# --pretrain_mm_mlp_adapter ./checkpoints/videollava-7b-pretrain/mm_projector.bin \

# CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 deepspeed
# --pretrain_mm_mlp_adapter ./checkpoints/videollava-7b-pretrain/mm_projector.bin \
CUDA_VISIBLE_DEVICES=0 TORCH_CPP_LOG_LEVEL=INFO NCCL_DEBUG=INFO deepspeed videollava/train/train_mem.py \
    --deepspeed ./scripts/zero2_offload.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path ${JSON_FOLDER}/train.json \
    --video_folder ${VIDEO_FOLDER} \
    --video_tower LanguageBind/LanguageBind_Video_merge \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/videollava-7b \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048  --tokenizer_model_max_length 3072 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "./cache_dir"