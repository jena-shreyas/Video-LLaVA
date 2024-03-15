#SBATCH -J vidllava_cvqa_ft
#SBATCH -p standard
#SBATCH -N 1
#SBATCH -n 1              
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH -t 0:15:0
#SBATCH --job-name=videollava_cvqa_ft
#SBATCH --error=videollava_cvqa_ft_%j.err
#SBATCH --output=videollava_cvqa_ft_%j.out
#SBATCH --mem=32G
module load conda
module load compiler/cuda/11.8
source activate videollava
ulimit -s unlimited

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

JSON_FOLDER="data/causalvidqa/annotations"
# IMAGE_FOLDER="llava_all_image_video"
VIDEO_FOLDER="data/causalvidqa/videos"

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed videollava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
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
    --output_dir ./checkpoints/videollava-7b-lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1024  --tokenizer_model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "./cache_dir"
