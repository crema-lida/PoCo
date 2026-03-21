source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch
tensorboard --logdir=runs --port 6006 &