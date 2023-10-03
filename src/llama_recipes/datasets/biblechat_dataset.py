from datasets import load_dataset
from llama_recipes.configs.datasets import biblechat_dataset

from gloohack.dataset.script import get_dataset
from gloohack.modelling.config import get_tokenizer


B_INST, E_INST = "[INST]", "[/INST]"
DATASET = "tr416/gloo_dataset_v2"
TOKENIZED_DATASET = "tr416/gloo_dataset_v2_tokenized"


def get_custom_dataset(dataset_config, tokenizer, split):
    ds = load_dataset(TOKENIZED_DATASET, split=split, use_auth_token=True)
    return ds

def load_custom_dataset(dataset_config, tokenizer, split):
    dataset = load_dataset(DATASET, split=split, use_auth_token=True)
    fmt_dataset = dataset.map(lambda x: format_dialog(x, tokenizer))
    fmt_dataset = fmt_dataset.remove_columns(['user', 'llm'])
    return fmt_dataset


def format_dialog(dialog_pair, tokenizer):
    fmt_pair = format_as_str(dialog_pair)
    dialog_tokens = tokenizer(fmt_pair,
                              padding='max_length',
                              max_length=512,
                              truncation=True)
    return dialog_tokens


def format_as_str(dialog_pair):
    fmt_str = f"{B_INST} {(dialog_pair['user']).strip()} {E_INST} {(dialog_pair['llm']).strip()} "
    return fmt_str


if __name__ == '__main__':
    tokenizer = get_tokenizer()
    ds = get_custom_dataset(biblechat_dataset(), tokenizer, "train")
