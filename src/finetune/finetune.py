from trainer import MultiTaskTrainer, PolymerDataset

if __name__ == '__main__':
    model_config = dict(
        input_dim=512,
        hidden_dim=512,
        num_hidden_layers=1,
        dropout=0.2,
    )

    encoder_path = './checkpoints/PoCo/final'
    output_name = 'MTL/PoCo'
    dataset = PolymerDataset.from_dir(
        dataset_dir='./datasets/MTL_Khazana',
        encoder_path=encoder_path,
    )

    trainer = MultiTaskTrainer(
        model_config, dataset,
        output_dir=f'./checkpoints/{output_name}',
        logging_dir=f'./runs/{output_name}',
        n_folds=5,
        n_trials=3,
        max_epochs=200,
        learning_rate=0.001,
        train_batch_size=32,
    )
    trainer.train()
