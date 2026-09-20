from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / 'output'
SOURCES = {
    'MTL': 'datasets/MTL_Khazana/pivot_table.csv',
    'RadonPy': 'datasets/RadonPy/PI1070.csv',
    'OPC': 'datasets/OPC/cleaned.csv',
}
MODELS = {
    'PoCo': dict(path=ROOT / 'checkpoints/PoCo/final', label='PoCo', color='#3F9AAE'),
    'MLM': dict(path=ROOT / 'checkpoints/PoCo-MLM/checkpoint-50000', label='MLM', color='#F9A66E'),
    'polyBERT': dict(path=ROOT / 'src/finetune/benchmark/polyBERT', label='polyBERT', color='#B08BBE'),
    'random': dict(path=ROOT / 'checkpoints/PoCo/final', label='Untrained', color='#9CA7AD'),
}
SAMPLE_SIZE = 1024
EVALUATION_SIZE = 500
BATCH_SIZE = 32
CPU_THREADS = 4
