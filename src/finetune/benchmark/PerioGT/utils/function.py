import numpy as np
import torch
import yaml


def load_config(args, file_path="../config.yaml"):
    with open(file_path, "r") as yaml_file:
        config_dict = yaml.safe_load(yaml_file)
    return config_dict[args.backbone][args.config]


def preprocess_batch_light(batch_num, batch_num_target, tensor_data):
    batch_num = np.concatenate([[0], batch_num], axis=-1)
    cs_num = np.cumsum(batch_num)
    add_factors = np.concatenate(
        [[cs_num[i]] * batch_num_target[i] for i in range(len(cs_num) - 1)],
        axis=-1,
    )
    return tensor_data + torch.from_numpy(add_factors).reshape(-1, 1)
