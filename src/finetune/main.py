from trainer import MultiTaskTrainer

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    dataset_config = dict(
        name='MTL',
        path='./datasets/MTL_Khazana/pivot_table.csv',
        # name='PolyOmics',
        # path='./datasets/PolyOmics/cleaned.csv',
    )
    model_config = dict(
        input_dim=512 * 3,
        hidden_dim=512,
        num_hidden_layers=1,
        dropout=0.2,
    )

    output_name = 'MTL/PoCo-final'
    encoder_path = './checkpoints/PoCo/final'

    trainer = MultiTaskTrainer(
        model_config, dataset_config,
        encoder_path=encoder_path,
        output_dir=f'./checkpoints/{output_name}',
        logging_dir=f'./runs/{output_name}',
        concat_last_layers=3,
        n_folds=5,
        n_trials=3,
        max_epochs=200,
        learning_rate=0.001,
        train_batch_size=32,
    )
    trainer.train()
    exit()

    output_name = 'MTL/PoCo-10k'
    encoder_dir = './checkpoints/PoCo-10k'
    all_checkpoints = [f'checkpoint-{step}' for step in range(10000, 50001, 10000)]

    for checkpoint in all_checkpoints:
        encoder_path = f'{encoder_dir}/{checkpoint}'

        trainer = MultiTaskTrainer(
            model_config, dataset_config,
            encoder_path=encoder_path,
            output_dir=f'./checkpoints/{output_name}/{checkpoint}',
            n_folds=5,
            n_trials=3,
            max_epochs=200,
            learning_rate=0.001,
            train_batch_size=32,
        )
        trainer.train()
