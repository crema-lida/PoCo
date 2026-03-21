import os
import sentencepiece as spm

elements = ['H', 'B', 'C', 'N', 'O', 'Si', 'P', 'S', 'F', 'Cl', 'Br', 'I']
aromatic = ['b', 'c', 'n', 'o', 'p', 's']

special_tokens = [
    "[*]",
    "(", ")", "=", "@", "#",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "-", "+",
    "/", "\\",
    "%", "[", "]",
]
special_tokens += elements + aromatic

input_path = 'datasets/PI1M/cleaned/train.csv'
output_dir = 'src/pretrain/tokenizer'

os.makedirs(output_dir, exist_ok=True)

spm.SentencePieceTrainer.train(
    input=input_path,
    model_prefix=f'{output_dir}/spm',
    model_type='bpe',
    vocab_size=len(special_tokens) + 4,
    add_dummy_prefix=False,
    user_defined_symbols=special_tokens,
    unk_piece='[UNK]',
    bos_piece='[CLS]',
    eos_piece='[SEP]',
)
