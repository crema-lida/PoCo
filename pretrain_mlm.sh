export OMP_NUM_THREADS=1
export PYTHONWARNINGS="ignore::FutureWarning"

DATASET_PATH=./datasets/PI1M/cleaned/train.csv
TOKENIZER_PATH=./src/pretrain/tokenizer

MODEL_NAME=PoCo-MLM-aug
torchrun --nproc-per-node 2 ./src/pretrain/pretrain_mlm.py \
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
        max_steps=100000 \
        per_device_train_batch_size=128 \
        gradient_accumulation_steps=8 \
