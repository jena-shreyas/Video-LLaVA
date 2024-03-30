FILENAME=$1

CKPT_NAME="videollava-7b"
model_path="LanguageBind/Video-LLaVA-7B"
cache_dir="./cache_dir"
GPT_Zero_Shot_QA="data/causalvidqa"
video_dir="${GPT_Zero_Shot_QA}/videos"
gt_file_question="${GPT_Zero_Shot_QA}/annotations/data_${FILENAME}.json"
date=$(date +"%d_%m_%Y_%H_%M_%S")
output_dir="results/eval/${CKPT_NAME}/causalvidqa/predictions/${FILENAME}"


gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 videollava/eval/video/run_inference_video_qa.py \
      --model_path ${model_path} \
      --cache_dir ${cache_dir} \
      --video_dir ${video_dir} \
      --gt_file_question ${gt_file_question} \
      --output_dir ${output_dir} \
      --output_name ${CHUNKS}_${IDX} \
      --num_chunks $CHUNKS \
      --chunk_idx $IDX &
done

wait

output_file=${output_dir}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${output_dir}/${CHUNKS}_${IDX}.json >> "$output_file"
done