GPU_LIST="0 1 2 3 4 5 6 7"  # GPU list for training, at least 8 48G GPUs are required
GPU_IDS=0,1,2,3,4,5,6,7
NAME="base"  # the name of the prompt
FDIR=PATH_TO_OUTPUT_DIR  # the path to the output directory
REF_MODEL_PT=PATH_TO_REF_MODEL  # the path to the reference model
JUDGE_MODEL_PT=$REF_MODEL_PT  # the path to the judge model
MODEL_TYPE=qwen  # the model type, can be qwen or llama
TOKENIZER_PT=Qwen/Qwen2.5-7B-Instruct  # the path to the tokenizer (Qwen/Qwen2.5-7B-Instruct or Llama/
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
INPUT_DIR=$FDIR

mkdir -p $FDIR
# generating samples
echo "Generating sample"
CUDA_VISIBLE_DEVICES=$GPU_IDS python sampling.py \
        --num_gpus 8 \
        --model_type $MODEL_TYPE \
        --model_pt $REF_MODEL_PT \
        --tokenizer_pt $TOKENIZER_PT \
        --num_samples 5  \
        --top_p 0.95 \
        --input_dir data/prompts/test.deepseek.v3.jsonl \
        --output_dir $FDIR/val.samples.jsonl \
        --gpuids $GPU_LIST \
        --num_workers 8

CUDA_VISIBLE_DEVICES=$GPU_IDS python sampling.py \
        --num_gpus 8 \
        --model_type $MODEL_TYPE \
        --model_pt $REF_MODEL_PT \
        --tokenizer_pt Qwen/Qwen2.5-7B-Instruct \
        --num_samples 5  \
        --top_p 0.95 \
        --input_dir data/prompts/train.deepseek.v3.jsonl \
        --output_dir $FDIR/train.samples.jsonl  \
        --gpuids $GPU_LIST \
        --num_workers 8

# scoring samples
echo "Scoring samples"

if [ "$NAME" == "base" ]; then
    CUDA_VISIBLE_DEVICES=$GPU_IDS python scoring.py \
        --src_dir data/prompts/test.deepseek.v3.jsonl \
        --input_dir $INPUT_DIR/val.samples.jsonl \
        --output_dir $FDIR/val.samples.pairs.jsonl \
        --gpuids $GPU_LIST \
        --model_pt $JUDGE_MODEL_PT \
        --batch_size 16 \
        --score_mode pairwise \
        --model_type pairwise-lm \
        --logprobs 5 \
        --prompt_dir prompts/${NAME}.txt \
        --num_workers 8 \
        --llm_type $MODEL_TYPE
else
    CUDA_VISIBLE_DEVICES=$GPU_IDS python scoring.py \
        --src_dir data/prompts/test.deepseek.v3.jsonl \
        --input_dir $INPUT_DIR/val.samples.jsonl \
        --output_dir $FDIR/val.samples.pairs.jsonl \
        --gpuids $GPU_LIST \
        --model_pt $JUDGE_MODEL_PT \
        --batch_size 16 \
        --score_mode pairwise \
        --model_type pairwise-lm \
        --logprobs 5 \
        --prompt_dir prompts/${NAME}.txt \
        --ref_dir data/v3_references/test.jsonl \
        --num_workers 8 \
        --llm_type $MODEL_TYPE
fi

if [ "$NAME" == "base" ]; then
    CUDA_VISIBLE_DEVICES=$GPU_IDS python scoring.py \
        --src_dir data/prompts/train.deepseek.v3.jsonl  \
        --input_dir $INPUT_DIR/train.samples.jsonl \
        --output_dir $FDIR/train.samples.pairs.jsonl \
        --gpuids $GPU_LIST \
        --model_pt $JUDGE_MODEL_PT \
        --batch_size 16 \
        --score_mode pairwise \
        --model_type pairwise-lm \
        --logprobs 5 \
        --prompt_dir prompts/${NAME}.txt \
        --num_workers 8 \
        --llm_type $MODEL_TYPE
else
    CUDA_VISIBLE_DEVICES=$GPU_IDS python scoring.py \
        --src_dir data/prompts/train.deepseek.v3.jsonl \
        --input_dir $INPUT_DIR/train.samples.jsonl \
        --output_dir $FDIR/train.samples.pairs.jsonl \
        --gpuids $GPU_LIST \
        --model_pt $JUDGE_MODEL_PT \
        --batch_size 16 \
        --score_mode pairwise \
        --model_type pairwise-lm \
        --logprobs 5 \
        --prompt_dir prompts/${NAME}.txt \
        --ref_dir data/v3_references/train.jsonl \
        --num_workers 8 \
        --llm_type $MODEL_TYPE
fi

# processing

echo "Processing"

python data_processing.py \
  --task make_output_pair_from_pm \
  --input_dir $FDIR/train.samples.pairs.jsonl \
  --output_dir $FDIR/train.pairs.jsonl \
  --num_workers 32 \
  --tokenizer_pt $TOKENIZER_PT  \
  --model_type $MODEL_TYPE \
  --pm_tokenizer_pt $TOKENIZER_PT


python data_processing.py \
  --task make_output_pair_from_pm \
  --input_dir $FDIR/val.samples.pairs.jsonl \
  --output_dir $FDIR/val.pairs.jsonl \
  --num_workers 32 \
  --tokenizer_pt $TOKENIZER_PT \
  --model_type $MODEL_TYPE \
  --pm_tokenizer_pt $TOKENIZER_PT


mkdir -p $FDIR/data
# get logprobs
echo "Getting logprobs using the latest model"
CUDA_VISIBLE_DEVICES=$GPU_IDS python get_logprobs.py \
    --input_dir $FDIR/train.pairs.jsonl \
    --gpuids $GPU_LIST \
    --output_dir $FDIR/data/train.jsonl \
    --model_type $MODEL_TYPE \
    --model_pt $REF_MODEL_PT \
    --tokenizer_pt $TOKENIZER_PT \
    --batch_size 4

CUDA_VISIBLE_DEVICES=$GPU_IDS python get_logprobs.py \
    --input_dir $FDIR/val.pairs.jsonl \
    --gpuids $GPU_LIST \
    --output_dir $FDIR/data/test.jsonl \
    --model_type $MODEL_TYPE \
    --model_pt $REF_MODEL_PT \
    --tokenizer_pt $TOKENIZER_PT \
    --batch_size 4

# training
for BETA in 0.01 0.005 0.02 0.05 0.2 
do
    # training with different beta values
    NEW_FDIR=${FDIR}-${BETA}
    mkdir -p $NEW_FDIR
    CUDA_VISIBLE_DEVICES=$GPU_IDS accelerate launch --mixed_precision bf16 \
        --num_machines 1 \
        --num_processes 8 \
        --use_deepspeed \
        --deepspeed_config_file deepspeed.conf \
        --main_process_port 29770 \
        dpo.py \
        --epoch 1 \
        --beta $BETA \
        --dataset $FDIR/data \
        --model_type $MODEL_TYPE \
        --exp_name $NEW_FDIR/ckpts  \
        --pretrained $REF_MODEL_PT \
        --gradient_checkpointing \
        --lr_schedule cosine \
        --accumulate_step 8 \
        --eval_interval 400 \
        --max_lr 0.0000005 \
        -l
done

