import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from llama_recipes.configs.datasets import biblechat_dataset

from gloohack.dataset.script import get_dataset
from gloohack.modelling.config import get_tokenizer


B_INST, E_INST = "[INST]", "[/INST]"
DATASET = "tr416/gloo_dataset_llama_format"
TOKENIZED_DATASET = "tr416/gloo_dataset_llama_format_tokenized"


def get_custom_dataset(dataset_config, tokenizer, split):
    hf_ds = load_dataset(TOKENIZED_DATASET, split=split, use_auth_token=True)
    torch_ds = huggingface_to_pytorch(hf_ds)
    return torch_ds


def load_custom_dataset(dataset_config, tokenizer, split):
    dataset = load_dataset(DATASET, split=split, use_auth_token=True)
    fmt_dataset = dataset.map(lambda x: tokenize_dialog(x, tokenizer))
    fmt_dataset = fmt_dataset.remove_columns(['user', 'llm'])
    return fmt_dataset


def tokenize_dialog(dialog_pair, tokenizer):
    dialog_str = f"{dialog_pair['user']} {dialog_pair['llm']}"
    dialog_tokens = tokenizer(dialog_str,
                              padding='max_length',
                              max_length=512,
                              truncation=True)
    return dialog_tokens


def huggingface_to_pytorch(hf_dataset):
    class CustomPyTorchDataset(Dataset):
        def __init__(self, hf_dataset):
            self.hf_dataset = hf_dataset

        def __len__(self):
            return len(self.hf_dataset)

        def __getitem__(self, idx):
            item = self.hf_dataset[idx]
            item['input_ids'] = torch.tensor(item['input_ids'])
            item['labels'] = torch.tensor(item['input_ids'])
            item['attention_mask'] = torch.tensor(item['attention_mask'])
            return item

    return CustomPyTorchDataset(hf_dataset)



if __name__ == '__main__':
    tokenizer = get_tokenizer()
    ds  = load_custom_dataset(None, tokenizer, 'test')
