export OMP_NUM_THREADS=1
export PYTHONWARNINGS="ignore::FutureWarning"

DATASET_PATH=./datasets/PI1M/cleaned/train.csv
TOKENIZER_PATH=./src/pretrain/tokenizer

MODEL_NAME=PoCo
torchrun --nproc-per-node 2 ./src/pretrain/train_transformer.py \
    --dataset-path $DATASET_PATH \
    --tokenizer-path $TOKENIZER_PATH \
    --model-config \
        hidden_size=512 \
        num_hidden_layers=3 \
        num_attention_heads=8 \
        intermediate_size=2048 \
    --training-args \
        output_dir=./checkpoints/$MODEL_NAME \
        logging_dir=./runs/$MODEL_NAME \
        max_steps=50000 \
        per_device_train_batch_size=128 \
        gradient_accumulation_steps=8 \
    --moco-config \
        tau_pos=0.05 \
        tau_neg=0.07 \
        queue_size=65536 \
        momentum=0.999 \
        refine_start_step=10000
