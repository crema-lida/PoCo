from trainer import MultiTaskTrainer, PolymerDataset

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    dataset_dir = './datasets/MTL_Khazana'
    # dataset_dir = './datasets/PolyOmics'
    model_config = dict(
        input_dim=512 * 3,
        hidden_dim=512,
        num_hidden_layers=1,
        dropout=0.2,
    )

    encoder_path = './checkpoints/PoCo/final'
    dataset = PolymerDataset.from_dir(
        dataset_dir,
        encoder_path=encoder_path,
        concat_last_layers=3,
    )
    output_name = f'{dataset.name}/PoCo-final'

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
    exit()

    encoder_dir = './checkpoints/PoCo-10k'
    all_checkpoints = [f'checkpoint-{step}' for step in range(10000, 50001, 10000)]

    for checkpoint in all_checkpoints:
        encoder_path = f'{encoder_dir}/{checkpoint}'
        dataset = PolymerDataset.from_dir(
            dataset_dir,
            encoder_path=encoder_path,
        )
        output_name = f'{dataset.name}/PoCo-10k'

        trainer = MultiTaskTrainer(
            model_config, dataset,
            output_dir=f'./checkpoints/{output_name}/{checkpoint}',
            n_folds=5,
            n_trials=3,
            max_epochs=200,
            learning_rate=0.001,
            train_batch_size=32,
        )
        trainer.train()
