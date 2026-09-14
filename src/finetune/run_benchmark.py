import multiprocessing as mp
from pathlib import Path


DATASETS = {
    'MTL': dict(dir='./datasets/MTL_Khazana', n_trials=3, batch_size=32),
    'RadonPy': dict(dir='./datasets/RadonPy', n_trials=3, batch_size=32),
    'OPC': dict(dir='./datasets/OPC', n_trials=3, batch_size=32),
    'Gas': dict(dir='./datasets/Gas', n_trials=3, batch_size=32),
    'PolyOmics': dict(dir='./datasets/PolyOmics', n_trials=1, batch_size=256),
}

ENCODERS = {
    'PoCo': dict(encoder_path='./checkpoints/PoCo/final', input_dim=512),
    'polyBERT': dict(encoder_path='polyBERT', input_dim=600),
    'TransPolymer': dict(encoder_path='TransPolymer', input_dim=768),
    'PolyCL': dict(encoder_path='PolyCL', input_dim=600, pooling='cls'),
    'MMPolymer': dict(encoder_path='MMPolymer', input_dim=1280, pooling='cls'),
    'PerioGT': dict(encoder_path='PerioGT', input_dim=2304),
}

OUTPUT_ROOT = Path('./checkpoints/downstream')
LOGGING_ROOT = Path('./runs/downstream')
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
              output_root, logging_root, model_config, training_config):
    from trainer import MultiTaskTrainer, PolymerDataset

    dataset_output = output_root / dataset_name
    n_trials = dataset_config['n_trials']
    tasks = dataset_config.get('tasks')
    embedding_cache = dataset_output / 'embeddings' / f'{encoder_name}.npy'
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
        experiment_name=encoder_name,
        logging_dir=str(logging_root / dataset_name / encoder_name),
        n_trials=n_trials,
        train_batch_size=dataset_config['batch_size'],
        **training_config,
    )
    trainer.train()


def main():
    context = mp.get_context('spawn')
    for dataset_name, dataset_config in DATASETS.items():
        for encoder_name, encoder_config in ENCODERS.items():
            process = context.Process(
                target=train_one,
                args=(dataset_name, dataset_config, encoder_name, encoder_config,
                      OUTPUT_ROOT, LOGGING_ROOT, MODEL_CONFIG, TRAINING_CONFIG),
            )
            process.start()
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(f'{dataset_name}/{encoder_name} exited with code {process.exitcode}')


if __name__ == '__main__':
    main()
