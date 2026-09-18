from pathlib import Path

from trainer import MultiTaskTrainer, PolymerDataset


if __name__ == '__main__':
    output_dir = Path('./checkpoints/downstream/MTL')
    model_config = dict(
        input_dim=512,
        hidden_dim=512,
        num_hidden_layers=1,
        dropout=0.2,
    )
    training_config = dict(
        n_folds=5,
        n_trials=3,
        seed=42,
        max_epochs=200,
        min_steps=50,
        early_stopping_patience=50,
        learning_rate=0.001,
        train_batch_size=32,
        inner_validation_size=0.2,
        validation_only=False,
        drop_last=False,
        resume=True,
    )
    dataset = PolymerDataset.from_dir(
        dataset_dir='./datasets/MTL_Khazana',
        encoder_path='./checkpoints/PoCo/final',
        embedding_cache=str(output_dir / 'embeddings/PoCo.npy'),
    )
    trainer = MultiTaskTrainer(
        model_config,
        dataset,
        output_dir=str(output_dir),
        experiment_name='PoCo',
        **training_config,
    )
    trainer.train()
