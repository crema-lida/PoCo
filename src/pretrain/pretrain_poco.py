import argparse
import ast

import torch
from transformers import DataCollatorForLanguageModeling
from transformers import DebertaV2Tokenizer, RoFormerConfig
from transformers import TrainingArguments

from poco import PolymerContrastModel
from trainer import MomentumContrastTrainer
from build_optimizers import build_optimizers
from build_dataset import build_dataset

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
        "--moco-config",
        nargs="*",
        default=None,
        help="Overrides parameters for MomentumContrastTrainer",
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
        position_embedding_type="rotary",  # set to "absolute" to ablate rotary embeddings
        pad_token_id=tokenizer.pad_token_id,
        proj_hidden_layers=1,
        proj_dim=256,
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
        report_to='tensorboard',
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

    def collate(batch):
        n_views = len(batch[0])
        views = [[] for _ in range(n_views)]
        for data in batch:
            for i, input_ids in enumerate(data.values()):
                views[i].append({'input_ids': input_ids})

        collated = [collator(v) for v in views]

        return {
            'input_ids_all': [v['input_ids'] for v in collated],
            'attention_mask_all': [v['attention_mask'] for v in collated],
        }

    model = PolymerContrastModel(model_config)

    optimizer, lr_scheduler = build_optimizers(
        model,
        base_lr=training_args.learning_rate,
        layer_decay=args.llrd,
        scheduler_type=training_args.lr_scheduler_type,
        warmup_steps=training_args.warmup_steps,
        max_steps=training_args.max_steps,
    )

    dataset_train = build_dataset(dataset_path, tokenizer, n_views=2)

    moco_config = dict(
        tau_pos=0.05,
        tau_neg=0.07,
        queue_size=65536,
        momentum=0.999,
    )
    moco_config.update(_parse_kv_list(args.moco_config))
    
    trainer = MomentumContrastTrainer(
        model=model,
        args=training_args,
        data_collator=collate,
        optimizers=(optimizer, lr_scheduler),
        train_dataset=dataset_train,
        processing_class=tokenizer,
        **moco_config,
    )
    trainer.train(resume_from_checkpoint=args.resume_checkpoint)

    output_dir = f'{training_args.output_dir}/final'
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
