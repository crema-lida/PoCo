from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / 'output'
SOURCES = {
    'MTL': 'datasets/MTL_Khazana/pivot_table.csv',
    'RadonPy': 'datasets/RadonPy/PI1070.csv',
    'OPC': 'datasets/OPC/cleaned.csv',
}
MODELS = {
    'PoCo': dict(path=ROOT / 'checkpoints/PoCo/final', label='PoCo', color='#087F8C'),
    'MLM': dict(path=ROOT / 'checkpoints/PoCo-MLM/checkpoint-50000', label='PoCo-MLM', color='#D28E61'),
    'random': dict(path=ROOT / 'checkpoints/PoCo/final', label='Untrained RoFormer', color='#9CA7AD'),
    'polyBERT': dict(path=ROOT / 'src/finetune/benchmark/polyBERT', label='polyBERT', color='#8174A5'),
}
SAMPLE_SIZE = 1024
VIEW_SAMPLE_SIZE = 256
BATCH_SIZE = 32
CPU_THREADS = 4
