import argparse
import ast

import torch
from transformers import DataCollatorForLanguageModeling
from transformers import DebertaV2Tokenizer, RoFormerConfig, RoFormerForMaskedLM
from transformers import TrainingArguments, Trainer
from datasets import load_dataset

import sys
from pathlib import Path
path = Path(__file__).resolve().parent
sys.path.append(path.parent.as_posix())

from build_optimizers import build_optimizers
from utils import augment_smiles

torch.set_float32_matmul_precision('high')

def _parse_value(raw):
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def _parse_kv_list(kv_list):
    overrides = {}
    if not kv_list:
        return overrides
    for item in kv_list:
        if "=" not in item:
            raise ValueError(f"config override must be key=value, got: {item}")
        key, value = item.split("=", 1)
        overrides[key] = _parse_value(value)
    return overrides


def _parse_args():
    parser = argparse.ArgumentParser(description="Pretrain PoCo model")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--resume-checkpoint", default=False, action='store_true')
    parser.add_argument(
        "--model-config",
        nargs="*",
        default=None,
        help="Overrides RoFormerConfig",
    )
    parser.add_argument(
        "--training-args",
        nargs="*",
        default=None,
        help="Overrides TrainingArguments",
    )
    parser.add_argument(
        "--llrd",
        default=0.9,
        help="Layer-wise learning rate decay factor"
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    dataset_path = args.dataset_path
    tokenizer_path = args.tokenizer_path

    tokenizer = DebertaV2Tokenizer.from_pretrained(tokenizer_path)
    collator = DataCollatorForLanguageModeling(
        tokenizer, mlm_probability=0.15, mask_replace_prob=1, random_replace_prob=0,
    )

    model_config = dict(
        vocab_size=len(tokenizer),
        hidden_size=512,
        num_hidden_layers=3,
        num_attention_heads=8,
        intermediate_size=2048,
        max_position_embeddings=384,
        pad_token_id=tokenizer.pad_token_id,
    )
    model_config.update(_parse_kv_list(args.model_config))
    model_config = RoFormerConfig(**model_config)
    
    training_args = dict(
        overwrite_output_dir=True,
        max_steps=50_000,
        per_device_train_batch_size=128,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        lr_scheduler_type='constant_with_warmup',
        warmup_steps=5000,
        logging_steps=50,
        save_steps=5000,
        fp16=True,
        dataloader_drop_last=True,
        dataloader_num_workers=16,
        dataloader_persistent_workers=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        ddp_broadcast_buffers=False,
        torch_compile=True,
    )
    training_args.update(_parse_kv_list(args.training_args))
    training_args = TrainingArguments(**training_args)

    tokenize_config = dict(
        truncation=True,
        max_length=384,
        return_token_type_ids=False,
        return_attention_mask=False,
    )

    def collate(batch):
        tokenized = tokenizer(
            [augment_smiles(item['SMILES']) for item in batch],
            **tokenize_config,
        )
        features = [{'input_ids': ids} for ids in tokenized['input_ids']]
        return collator(features)

    model = RoFormerForMaskedLM(model_config)

    optimizer, lr_scheduler = build_optimizers(
        model,
        base_lr=training_args.learning_rate,
        layer_decay=args.llrd,
        scheduler_type=training_args.lr_scheduler_type,
        warmup_steps=training_args.warmup_steps,
        max_steps=training_args.max_steps,
    )

    dataset_train = load_dataset(
        path='csv',
        data_files={'train': dataset_path},
        column_names=['SMILES'],
    )['train']
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate,
        optimizers=(optimizer, lr_scheduler),
        train_dataset=dataset_train,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=args.resume_checkpoint)

    output_dir = f'{training_args.output_dir}/final'
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
