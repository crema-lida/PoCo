import multiprocessing as mp
from pathlib import Path


DATASETS = {
    'MTL': dict(dir='./datasets/MTL_Khazana', n_trials=3, batch_size=32),
}


ENCODERS = {
    'PoCo-w_gram': dict(encoder_path='./checkpoints/PoCo', input_dim=512),
    'PoCo-wo_gram': dict(encoder_path='./checkpoints/PoCo-wo_gram', input_dim=512),
    'PoCo-MLM': dict(encoder_path='./checkpoints/PoCo-MLM', input_dim=512),
}

CHECKPOINTS = range(0, 50001, 5000)


OUTPUT_ROOT = Path('./checkpoints/ablation')
MODEL_CONFIG = dict(hidden_dim=512, num_hidden_layers=1, dropout=0.2)
TRAINING_CONFIG = dict(
    n_folds=5,
    seed=42,
    max_epochs=200,
    min_steps=50,
    early_stopping_patience=50,
    learning_rate=0.001,
    inner_validation_size=0.2,
    drop_last=False,
    resume=True,
)


def train_one(dataset_name, dataset_config, encoder_name, encoder_config,
              output_root, model_config, training_config, checkpoint=None):
    from trainer import MultiTaskTrainer, PolymerDataset

    dataset_output = output_root / dataset_name
    n_trials = dataset_config['n_trials']
    tasks = dataset_config.get('tasks')
    if checkpoint is not None:
        embedding_cache = dataset_output / 'embeddings' / encoder_name / f'checkpoint-{checkpoint}.npy'
        experiment_name = f'{encoder_name}/checkpoint-{checkpoint}'
    else:
        embedding_cache = dataset_output / 'embeddings' / f'{encoder_name}.npy'
        experiment_name = encoder_name
    dataset = PolymerDataset.from_dir(
        dataset_config['dir'],
        encoder_path=encoder_config['encoder_path'],
        pooling=encoder_config.get('pooling', 'mean'),
        embedding_cache=str(embedding_cache),
    )
    trainer = MultiTaskTrainer(
        dict(model_config, input_dim=encoder_config['input_dim']),
        dataset,
        tasks=tasks,
        output_dir=str(dataset_output),
        experiment_name=experiment_name,
        n_trials=n_trials,
        train_batch_size=dataset_config['batch_size'],
        **training_config,
    )
    trainer.train()


def main():
    context = mp.get_context('spawn')
    for dataset_name, dataset_config in DATASETS.items():
        for encoder_name, encoder_config in ENCODERS.items():
            for ckpt in CHECKPOINTS:
                checkpoint_config = dict(
                    encoder_config,
                    encoder_path=str(Path(encoder_config['encoder_path']) / f'checkpoint-{ckpt}'),
                )
                process = context.Process(
                    target=train_one,
                    args=(dataset_name, dataset_config, encoder_name, checkpoint_config,
                          OUTPUT_ROOT, MODEL_CONFIG, TRAINING_CONFIG, ckpt),
                )
                process.start()
                process.join()
                if process.exitcode != 0:
                    raise RuntimeError(f'{dataset_name}/{encoder_name} exited with code {process.exitcode}')


if __name__ == '__main__':
    main()
