from trainer import MultiTaskTrainer

if __name__ == '__main__':
    datasets = [
        dict(
            name='MTL',
            path='./datasets/MTL_Khazana/pivot_table.csv',
            n_trials=3,
            batch_size=32,
        ),
        dict(
            name='PolyOmics',
            path='./datasets/PolyOmics/cleaned.csv',
            n_trials=1,
            batch_size=256,
        ),
    ]
    model_config = dict(
        hidden_dim=512,
        num_hidden_layers=1,
        dropout=0.2,
    )
    encoders = [
        dict(output_name='polyBERT', encoder_path='polyBERT', input_dim=600),
        dict(output_name='TransPolymer', encoder_path='TransPolymer', input_dim=768),
        dict(output_name='PolyCL', encoder_path='PolyCL', input_dim=600, pooling='cls'),
        dict(output_name='MMPolymer', encoder_path='MMPolymer', input_dim=1280, pooling='cls'),
        dict(output_name='PerioGT', encoder_path='PerioGT', input_dim=2304),
        dict(output_name='PoCo', encoder_path='./checkpoints/PoCo/final', input_dim=512),
        dict(output_name='PoCo_concat', encoder_path='./checkpoints/PoCo/final', input_dim=512 * 3, concat_last_layers=3),
    ]

    for dataset_config in datasets:
        for enc in encoders:
            output_name = f"{dataset_config['name']}/{enc['output_name']}"
            model_config['input_dim'] = enc['input_dim']
            trainer = MultiTaskTrainer(
                model_config, dataset_config,
                encoder_path=enc['encoder_path'],
                pooling=enc.get('pooling', 'mean'),
                concat_last_layers=enc.get('concat_last_layers', None),
                output_dir=f'./checkpoints/{output_name}',
                logging_dir=f'./runs/{output_name}',
                n_folds=5,
                n_trials=dataset_config['n_trials'],
                max_epochs=200,
                learning_rate=0.001,
                train_batch_size=dataset_config['batch_size'],
            )
            trainer.train()
